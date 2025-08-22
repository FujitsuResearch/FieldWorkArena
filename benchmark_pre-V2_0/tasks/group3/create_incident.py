#from browsergym.workarena.tasks.compositional.base import CompositionalTask, HumanEvalTask

from browsergym.workarena.instance import SNowInstance
from browsergym.workarena.tasks.form import CreateIncidentTask, GenericNewRecordTask
from browsergym.workarena.tasks.base import AbstractServiceNowTask

from browsergym.workarena.tasks.utils.utils import prettyprint_enum, check_url_suffix_match
from browsergym.workarena.api.utils import table_api_call


from playwright.sync_api._generated import Page
import logging
import playwright
from typing import Tuple
from browsergym.workarena.config import EXPECTED_INCIDENT_FORM_FIELDS_PATH

class CreateIncidentWithRetrievedInfoTask(GenericNewRecordTask):
    expected_fields_path = EXPECTED_INCIDENT_FORM_FIELDS_PATH
    def __init__(
        self,
        seed: int = None,
        instance=None,
        fixed_config: dict = None,
        check_record_created=True,
        **kwargs,
    ) -> None:
        super().__init__(
            seed=seed,
            instance=instance,
            form_url="/now/nav/ui/classic/params/target/incident.do",
            table_label="incident",
            prohibited_fields=["state"],
            fixed_config=fixed_config,
            check_record_created=check_record_created,
        )
        self.__dict__.update(kwargs)

        self.task_description = "Create an incident with the retrieved data"
        self.short_description = "Create an incident with the retrieved data"
        

    def setup_goal(self, page: Page) -> tuple[str, dict]:
        super(GenericNewRecordTask, self).setup_goal(page)

        assert self.all_configs is not None, "No configuration available for the task."
        config = self.fixed_config if self.fixed_config else self.random.choice(self.all_configs)
        # If fixed_config is not None we already set the required attributes in the constructor
        if self.fixed_config is None:
            self._set_required_config_attributes(config)
        self.protected_fields = self.task_fields
        # Generate the goal

        self.record_sys_id = self.template_record["number"]

        goal = (
            f"If you think it is necessary, create a new {self.table_label} with "
            + prettyprint_enum(
                [
                    f'a value of "{self.template_record[f]}"'
                    + f' for field "{config["fields"][f]}"'
                    for f in self.template_fields
                ]
            )
            + " and a value retrieved from the previous task"
            + prettyprint_enum(
                [
                    f' for field "{config["fields"][f]}"' for f in self.retrieve_fields
                ]
            )
            + ". Also upload the file in the attachment field. " 
            + "otherwise, do report_infeasible()."
        )

        #print(goal)
        info = {}

        return goal, info

    def _set_required_config_attributes(self, config: dict) -> None:
        """
        Set the required attributes for the task configuration.
        """
        # XXX Warning: Some subclasses may expect a specific order of elements
        self.template_record = config["template_record"]
        print(self.unique_valued_fields)
        for f, func in self.unique_valued_fields.items():
            self.template_record[f] = func(self.template_record[f])
        self.task_fields = config["task_fields"]
        if "retrieve_fields" not in config:
            config["retrieve_fields"] = []
        self.retrieve_fields = config["retrieve_fields"]
        self.template_fields = list(set(self.task_fields) - set(self.retrieve_fields))
        #self.expected_fields = self.task_fields
        
    def _page_on_right_url(self, page: Page) -> bool:
        """Checks if the page is on the right URL for validation + sets the page_on_form_view attribute"""
        page.wait_for_load_state("domcontentloaded")
        # Always take a minte to wait for the page to load
        #self._wait_for_ready(page, iframe_only=True)
        # check that the page is at the right url
        list_url = self.start_url.replace(".do", "_list.do")  # list view of records
        # Check whether we are in the form or list view
        self.page_is_form_view = check_url_suffix_match(
            page, expected_url=self.start_url, task=self
        )
        page_is_list_view = check_url_suffix_match(page, expected_url=list_url, task=self)
        right_url = self.page_is_form_view or page_is_list_view

        return right_url


    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:
        """
        Caveat: we check only if the expected fields have the right value. We don't Check
                if there are extra fields that shouldn't be there. We could have issues
                matching other fields since calculation rules may have changed through time.
                Maybe we should assign a random value from our list of choices to the fields
                that are not part of the task.

        """
        right_url = self._page_on_right_url(page)
        if not right_url:
            return (
                0,
                False,
                "",
                {
                    "message": f"The page is not in the right URL to validate task {self.__class__.__name__}."
                },
            )
        protected_field_changed = page.evaluate(
            "() => window.gsft_main.WORKARENA_BAD_FIELD_CHANGED"
        )
        if protected_field_changed:
            return (
                0,
                True,
                "",
                {"message": "Some fields outside of the task scope have been changed."},
            )
        if self.table_metadata is None and self.page_is_form_view:
            # XXX We need to ensure the table metadata as well as fields are set
            # before we can proceed with the cheat function
            self._wait_for_ready(page, iframe_only=True)
            self._get_form(page)
        if self.fields is None and self.page_is_form_view:
            self._get_fields(page)
        # Retrieve the created record's sys_id from the session storage
        sys_id = page.evaluate("localStorage").get(self.session_sys_id_field, None)
        # Check that a record has actually been created
        if sys_id is None:
            logging.info("No record has been created.")
            return (
                0,
                False,
                "",
                {"message": "The form has not been submitted."},
            )

        # Add the sysid to the list of created sysids
        # This is used to clean up the database after the task is completed.
        self.created_sysids.append(sys_id)
        # Pull the record from the database
        # XXX: It's possible that the record is not found, e.g., if form submission was rejected due to client-side
        #      validation errors. In this case, we should not raise an error and simply consider that no record was
        #      created. This is non-terminal for the task.
        record = table_api_call(
            instance=self.instance,
            table=self.table_name,
            params={
                "sysparm_query": f"sys_id={sys_id}",
                "sysparm_display_value": True,
            },
            wait_for_record=True,
            max_retries=20,  # Wait up to 10 seconds
            raise_on_wait_expired=False,
        )["result"]

        # This can happen if the form was submitted but was rejected due to invalid inputs (e.g., missing mandatory fields)
        if len(record) == 0:
            logging.info(
                "The record was not found in the database. Perhaps the form was not submitted correctly. "
                + sys_id,
            )
            return (
                0,
                False,
                "",
                {
                    "message": "The record was not found in the database. Perhaps the form was not submitted correctly."
                },
            )
        # Extract display values for reference fields
        record = {
            f: v if not isinstance(v, dict) else v["display_value"] for f, v in record[0].items()
        }
        # Check that the record matches the expected values
        for f in self.task_fields:
            if f in self.retrieve_fields:
                continue

            if record[f] != self.template_record[f]:
                logging.info(
                    f'The field "{self.fields[f]["label"]}" has the wrong value. Expected: "{self.template_record[f]}", got: "{record[f]}".'
                )
                error_msg = f'The field "{self.fields[f]["label"]}" has the wrong value.'
                return (
                    0,
                    True,  # End episode (incorrect information pushed to the DB)
                    error_msg,
                    {"message": error_msg},
                )

        return (
            1,
            True,
            "Nice work, thank you!",
            {"message": "The record was successfully created."},
        )

    # def teardown(self):
    #     # only use for debug
    #     return 
    

if __name__ == "__main__":
    print("ok")
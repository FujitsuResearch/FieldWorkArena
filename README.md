# FieldWorkArena

## Overview

The introduction of AI agents is being considered to address the challenges faced by many workplaces, such as the aging of the population, lack of human resources, and delays in decision-making. In order to improve the functionality of AI agents, we have developed and provided a benchmark suite to evaluate AI agents by extending the evaluation method of web operations to field operations.

FieldWorkArena is a groundbreaking benchmark suite for evaluating AI agents. By using data and tasks from Fujitsu's actual factories and warehouses, we quantitatively evaluate how effectively AI agents work in the field. This clarifies the challenges of AI adoption and ensures evidence when applied in the field.

See below for more details. \
https://en-documents.research.global.fujitsu.com/fieldworkarena/

## Getting Started
The current reporting functionality of FieldWorkArena utilizes
[Browsergym](https://github.com/ServiceNow/BrowserGym) and [WorkArena](https://github.com/ServiceNow/WorkArena). Therefore, it is necessary to use ServiceNow instance in this implementation. \
In the future, the implementation may change in line with modification to the action space and task definitions.

### Create ServiceNow Instance
1. Go to https://developer.servicenow.com/ and create an account.
2. Click on `Request an instance` and select the `Washington` release (initializing the instance will take a few minutes),
   If you can't select release, once you request an instance for default release, do `Release instance `and click `Request an instance` again.
3. Once the instance is ready, you should see your instance URL and credentials. If not, click _Return to the Developer Portal_, then navigate to _Manage instance password_ and click _Reset instance password_.
4. You should now see your URL and credentials. Based on this information, set the following environment variables:
    * `SNOW_INSTANCE_URL`: The URL of your ServiceNow developer instance
    * `SNOW_INSTANCE_UNAME`: The username, should be "admin"
    * `SNOW_INSTANCE_PWD`: The password, make sure you place the value in quotes "" and be mindful of [escaping special shell characters](https://onlinelinuxtools.com/escape-shell-characters). Running `echo $SNOW_INSTANCE_PWD` should print the correct password.
5. Log into your instance via a browser using the admin credentials. Close any popup that appears on the main screen (e.g., agreeing to analytics).

**Warning:** Feel free to look around the platform, but please make sure you revert any changes (e.g., changes to list views, pinning some menus, etc.) as these changes will be persistent and affect the benchmarking process.

### Install FieldWorkArena and  Initialize your instance

```bash
git clone https://github.com/FujitsuResearch/FieldWorkArena.git
cd FieldWorkArena
pip install -r requirements.txt
pip install .
```
Then, install Playwright
```bash
playwright install
```
Finally, run this command in a terminal to upload the benchmark data to your ServiceNow instance:
```
workarena-install
```

### Download dataset 
1. Go to https://en-documents.research.global.fujitsu.com/fieldworkarena/ .
2. Click link on `評価用データ一式(Evaluation dataset)` and apply from Forms page,
3. Confirm the download URL in email sent from FieldWorkArena. (It may take a few business days.)
4. Unzip downloaded file. The files should be organized in the following directory structure:
```
FieldWorkArena \
├── ...\
├── data\
│   ├── document \
│   ├── image\
│   └── movie\
└── ...
```
## Use Sample Agent

### OpenAI API setting (for demo agent)
set environment variable
* `OAI_API_KEY `: You OpenAI API key 

### Demo
In these demos, the tasks is to search for incidents in the image according to the query and to report any incidents found.
```
python demo/run_demo.py --task_name fieldworkarena.demo.1.report 
python demo/run_demo.py --task_name fieldworkarena.demo.2.report 
python demo/run_demo.py --task_name fieldworkarena.demo.3.report 
python demo/run_demo.py --task_name fieldworkarena.demo.4.report 
```

### Benchmark
Run the following script, the results will be saved in the `results` directory.
#### Linux 
```
bash run_all_tasks.sh
```
#### Windows
```
.\run_all_tasks.bat
```
## Test Your Agent 
### Edit Agent
Agent is defined in 'demo/agent.py'.
For testing your agent, you should mainly modify 'get_action()' method.

### Submit Your Result
Compress the `results` directory and reply it to the email address with the download URL of the evaluation data .


## Inquiries and Support

To submit an inquiry, please follow these steps:

1. Visit [our page](https://en-documents.research.global.fujitsu.com/fieldworkarena/)
2. Click the "Inquiry" button on the bottom.
3. Fill out the form completely and accurately.

It may take a few business days to reply.

## Acknowledment
This implementation was created with reference to the source code for WorkArena, developed by ServiceNow Research.
- github: https://github.com/ServiceNow/WorkArena
- arxiv: 
    * [WorkArena](https://arxiv.org/pdf/2403.07718)
    * [WorkArena++](https://arxiv.org/pdf/2407.05291)

## Trouble shooting 
When the browser launched and the proxy auth dialog blocks the startup, please install chrome extension "Proxy Helper". After that, fill PAC URL and your account/password.

https://chromewebstore.google.com/detail/%E3%83%97%E3%83%AD%E3%82%AD%E3%82%B7%E3%83%BC%E3%83%98%E3%83%AB%E3%83%91%E3%83%BC/mnloefcpaepkpmhaoipjkpikbnkmbnic?hl=ja&pli=1


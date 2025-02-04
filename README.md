# FieldWorkArena

## Getting Started

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

### Download dataset (Currently only avaiable in Japan. Global release coming soon.)
1. Go to https://documents.research.global.fujitsu.com/fieldworkarena/ .
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
bash run_demo_group2.sh
```
#### Windows
```
.\run_demo_group2.bat
```
## Test Your Agent 
### Edit Agent
Agent is defined in 'demo/agent.py'.
For testing your agent, you should mainly modify 'get_action()' method.

### Submit Your Result
Compress the `results` directory and reply it to the email address with the download URL of the evaluation data .

## Trouble shooting 
When the browser launched and the proxy auth dialog blocks the startup, please install chrome extension "Proxy Helper". After that, fill PAC URL and your account/password.

https://chromewebstore.google.com/detail/%E3%83%97%E3%83%AD%E3%82%AD%E3%82%B7%E3%83%BC%E3%83%98%E3%83%AB%E3%83%91%E3%83%BC/mnloefcpaepkpmhaoipjkpikbnkmbnic?hl=ja&pli=1


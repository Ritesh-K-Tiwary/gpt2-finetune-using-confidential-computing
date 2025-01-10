# gpt2-finetune-using-confidential-computing

First login to google cloud then follow the given steps:

# Setup environment

Start with setting up the Google Cloud Confidential VM

## Step 1: Enable Required APIs
Open Google Cloud Console:

Navigate to the Google Cloud Console.
Enable APIs:

Go to APIs & Services > Library.
Enable the following APIs:
Compute Engine API
Cloud Storage API

## Step 2: Create a Confidential VM
Navigate to Compute Engine:

Go to Compute Engine > VM Instances in the Google Cloud Console.
Create a New Instance:

Click Create Instance.
Under Machine Configuration, select:
Series: N2D
Machine Type: n2d-standard-4 (or higher, depending on your requirements)
Check Confidential Computing.
Select AMD SEV to enable Confidential VM.
Choose the Operating System:

Under Boot Disk, click Change.
Select Ubuntu 20.04 LTS or Ubuntu 22.04 LTS.
Configure Firewall:

Under Firewall, check both:
Allow HTTP traffic
Allow HTTPS traffic
Create the VM:

Click Create to launch your Confidential VM.

## Step 3: Connect to Your VM
Go to Compute Engine > VM Instances.
Find your instance and click SSH to connect to it.

It will launch an execution environment

# Fine-tune gpt2 model:

model details is as:
 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

and dataset details is as:

from datasets import load_dataset

ds = load_dataset("fka/awesome-chatgpt-prompts")

datasets has 2 columns: act and prompt which is already splitted in train-test.


coding steps:

### Check Disk Usage

Before proceeding, identify which directories are consuming the most space.

Run the following command to analyze disk usage:

```bash
sudo du -sh /* | sort -h
```
This will show the size of each directory in your root filesystem.

###  Clean Up Unnecessary Files

Remove Old Python Environments or Unused Files
If you've installed virtual environments or unused files, delete them:

```bash
rm -rf ~/.cache/pip
rm -rf ~/venv  # Replace with the name of your virtual environment, if any
```

### Clean Up APT Cache

Free up space taken by APT package manager cache:

```bash
sudo apt-get clean
sudo apt-get autoremove -y
```

### Remove Snap Packages (Optional)

Snap packages like lxd and google-cloud-cli are consuming disk space. If you don't need them:

```bash
sudo snap remove lxd
sudo snap remove google-cloud-cli
```

## Create a New Python Virtual Environment

Install venv (if not already installed)
```bash
sudo apt-get update
sudo apt-get install python3-venv
```

### Create a New Virtual Environment
```bash
python3 -m venv gpt2_env
source gpt2_env/bin/activate
```

Upgrade Pip
Once the virtual environment is activated, upgrade pip to the latest version:

```bash
pip install --upgrade pip
```

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip git build-essential
pip3 install torch transformers cryptography datasets accelerate
```
# Finetuning GPT-2 Model

## Import Dependencies and Load Dataset

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("fka/awesome-chatgpt-prompts")

# Inspect the dataset
print(dataset)
print(dataset['train'].column_names)

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
```

## Preprocess the Dataset

We will preprocess the dataset to tokenize the prompt column and add the act column as metadata.

```python
def preprocess_data(examples):
    # Combine "act" and "prompt" for better fine-tuning
    input_texts = [f"{examples['act']}:\n{examples['prompt']}" for act, prompt in zip(examples['act'], examples['prompt'])]
    # Tokenize the input
    inputs = tokenizer(input_texts, truncation=True, padding="max_length", max_length=512)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

# Inspect tokenized dataset
print(tokenized_dataset['train'][0])
```

## Training:

### Define Training Arguments
Configure the training process, including output directory, batch size, number of epochs, etc.

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    push_to_hub=False,
    report_to="none",
    fp16=True,  # Use mixed precision for better performance
)
```

### Set Up the Trainer

Define the Trainer object for the fine-tuning process.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)
```

### Fine-Tune the Model
Run the fine-tuning process.

```python
trainer.train()
```

### Save the Fine-Tuned Model

After fine-tuning, save the model and tokenizer locally.
```python
trainer.save_model("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
```

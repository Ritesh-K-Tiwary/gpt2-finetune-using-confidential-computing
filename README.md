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
 ```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
```
and dataset details is as:
```python
from datasets import load_dataset

ds = load_dataset("fka/awesome-chatgpt-prompts")
```
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
tokenizer.pad_token = tokenizer.eos_token

# Split the dataset into train and test
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

def preprocess_data(examples):
    # Combine "act" and "prompt" for better fine-tuning
    input_texts = [f"{examples['act']}:\n{examples['prompt']}" for act, prompt in zip(examples['act'], examples['prompt'])]
    # Tokenize the input
    inputs = tokenizer(input_texts, truncation=True, padding="max_length", max_length=512)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

# Tokenize the dataset
tokenized_dataset = split_dataset.map(preprocess_data, batched=True, remove_columns=split_dataset['train'].column_names)

# Inspect tokenized dataset
print(tokenized_dataset['train'][0])
print(tokenized_dataset['test'][0])
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

# Encryption and Decryption

## Install Required Library
Make sure the cryptography library is installed in your environment:

```bash
pip install cryptography
```

## Encrypt the Model

This function will encrypt all the files in your model directory.

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def encrypt_model(model_dir, output_dir, key):
    """
    Encrypt all files in the model directory and save them to the output directory.
    Args:
        model_dir (str): Path to the directory containing the model files.
        output_dir (str): Path to save encrypted files.
        key (bytes): AES encryption key (32 bytes for AES-256).
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(model_dir):
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_dir, file + ".enc")
            # Generate a random Initialization Vector (IV)
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            # Read the file and encrypt it
            with open(input_file, 'rb') as f:
                plaintext = f.read()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            # Save the IV and ciphertext to the output file
            with open(output_file, 'wb') as f:
                f.write(iv + ciphertext)
```

## Decrypt the Model
This function will decrypt the previously encrypted model files.

```python
def decrypt_model(encrypted_dir, output_dir, key):
    """
    Decrypt all files in the encrypted directory and save them to the output directory.
    Args:
        encrypted_dir (str): Path to the directory containing encrypted files.
        output_dir (str): Path to save decrypted files.
        key (bytes): AES decryption key (same key used for encryption).
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(encrypted_dir):
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_dir, file.replace(".enc", ""))
            with open(input_file, 'rb') as f:
                # Extract the IV (first 16 bytes)
                iv = f.read(16)
                ciphertext = f.read()
            cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            # Decrypt the ciphertext
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            with open(output_file, 'wb') as f:
                f.write(plaintext)
```

## Generate a Secure Encryption Key
Generate a 32-byte (256-bit) key for AES encryption:

```python
# Generate a random 32-byte key for AES-256
key = os.urandom(32)

# Save the key securely (e.g., write it to a secure storage or display it for manual use)
print("Encryption Key:", key)
```

## Encrypt Your Model
Run the encryption function on your fine-tuned GPT-2 model directory:

```python
model_directory = "./fine_tuned_gpt2"  # Path to your model files
encrypted_directory = "./encrypted_model"  # Where encrypted files will be saved

encrypt_model(model_directory, encrypted_directory, key)
print("Model encrypted successfully!")
```

## Decrypt Your Model
Run the decryption function to restore your model:

```python
decrypted_directory = "./decrypted_model"  # Where decrypted files will be saved

decrypt_model(encrypted_directory, decrypted_directory, key)
print("Model decrypted successfully!")
```
## Verify Your Results
To verify the decryption process:

Compare the original model files with the decrypted files to ensure they match.
Try loading the decrypted model using the same code you used to load the original model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(decrypted_directory)
model = AutoModelForCausalLM.from_pretrained(decrypted_directory)
print("Decrypted model loaded successfully!")
```

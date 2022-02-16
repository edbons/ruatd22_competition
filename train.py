import pandas as pd
import numpy as np
import torch
import pickle
import os
from tqdm import tqdm
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, AutoConfig)
import datasets
from datasets import Dataset, load_metric, Features, ClassLabel, Value

print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)

def build_dataset(data: pd.DataFrame, tokenizer: AutoTokenizer, max_length=512, with_label=True):
    if with_label:
        class_names = ["M", "H"]
        features = Features({'Text': Value('string'), 'label': ClassLabel(names=class_names, num_classes=2)})
        dataset = Dataset.from_pandas(data, preserve_index=False, features=features)
        dataset = dataset.map(lambda e: tokenizer(e['Text'], truncation=True, padding='max_length', max_length=max_length), batched=True)    
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    else:
        dataset = Dataset.from_pandas(data, preserve_index=False)
        dataset = dataset.map(lambda e: tokenizer(e['Text'], truncation=True, padding='max_length', max_length=max_length), batched=True)    
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    return dataset

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    df_train = pd.read_csv('train.csv')
    df_val = pd.read_csv('val.csv')

    df_val.loc[df_val['Class'] == 'M', 'label'] = 0
    df_val.loc[df_val['Class'] == 'H', 'label'] = 1
    df_val = df_val.convert_dtypes()
    print(df_val.info())

    df_train.loc[df_train['Class'] == 'M', 'label'] = 0
    df_train.loc[df_train['Class'] == 'H', 'label'] = 1
    df_train = df_train.convert_dtypes()
    print(df_train.info())

    # model_name = "DeepPavlov/rubert-base-cased-sentence"
    model_name = "DeepPavlov/rubert-base-cased"
    print(model_name)
    # model_name = 'cointegrated/rubert-tiny'
    num_labels = 2
    batch_size = 32 # T4: 32, 
    epochs = 3
    lr = 2e-5
    max_len = 200

    tokenizer_bert = AutoTokenizer.from_pretrained(model_name)


    ds_val = build_dataset(df_val, tokenizer_bert, max_length=max_len)
    ds_train = build_dataset(df_train, tokenizer_bert, max_length=max_len)

    config_bert = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels)

    bert = AutoModelForSequenceClassification.from_pretrained(model_name, config=config_bert)

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments("test_trainer", 
                                    per_device_train_batch_size=batch_size, 
                                    per_device_eval_batch_size=batch_size,
                                    num_train_epochs=epochs,
                                    learning_rate=lr,
                                    save_strategy='epoch',
                                    evaluation_strategy='epoch',
                                    save_total_limit=2,
                                    load_best_model_at_end=True,
                                    do_train=True,
                                    do_eval=True,
                                    optim='adamw_torch',
                                    report_to="none"
                                    )

    trainer = Trainer(model=bert, 
                    args=training_args, 
                    train_dataset=ds_train, 
                    eval_dataset=ds_val,
                    compute_metrics=compute_metrics,
                    tokenizer=tokenizer_bert, 
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
                    )


    train_result = trainer.train()
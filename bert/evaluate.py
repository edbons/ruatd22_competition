import pandas as pd
import numpy as np
import os
import argparse
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer)
from train import build_dataset


def main(args: argparse.Namespace):
    os.chdir(os.path.dirname(__file__))
    df_test = pd.read_csv('test.csv')
    batch_size = args.n_batch 
    max_len = 200
    model_name_or_path = args.model_name_or_path
    if args.max_samples is not None:
        df_test = df_test[:args.max_samples]

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    ds_test = build_dataset(df_test, tokenizer, max_length=max_len, with_label=False)

    training_args = TrainingArguments("eval_trainer", 
                                per_device_train_batch_size=batch_size, 
                                per_device_eval_batch_size=batch_size,
                                do_train=False,
                                do_eval=False,
                                report_to="none"
                                )
    
    trainer = Trainer(model=model, args=training_args)
    predictions = trainer.predict(ds_test).predictions
    predictions = np.argmax(predictions, axis=1)
    df_test['label'] = predictions
    df_test.loc[df_test['label'] == 0, 'Class'] = 'M'
    df_test.loc[df_test['label'] == 1, 'Class'] = 'H'

    df_test.to_csv('submission.csv', columns=['Id','Class'], index=False) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--model_name_or_path', type=str, default='savedir', help='directory to save logs and checkpoints to')
    args = parser.parse_args()
    print(args)
    main(args)
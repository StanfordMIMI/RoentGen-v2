import pandas as pd
from torch.utils.data import Dataset

###########################################
class UnetInferenceDatasetCSV(Dataset):
    def __init__(self, csv_file, tokenizer) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        sample = {}
        sample["prompt_text"] = row["prompt_variation"]
        sample["stem"] = row["dicom_id"]

        prompt_tokenized = self.tokenizer(
            sample["prompt_text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        sample["prompt_ids"] = prompt_tokenized.input_ids

        uncond_tokenized = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        sample["uncond_ids"] = uncond_tokenized.input_ids

        return sample

    def collate_fn_inference(self, examples):
        input_ids = [example["prompt_ids"] for example in examples]
        uncond_ids = [example["uncond_ids"] for example in examples]

        padded_tokens = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        )
        padded_uncond = self.tokenizer.pad(
            {"input_ids": uncond_ids}, padding=True, return_tensors="pt"
        )

        stem_values = [example["stem"] for example in examples]
        prompt_texts = [example["prompt_text"] for example in examples]

        return {
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
            "uncond_ids": padded_uncond.input_ids,
            "stem_values": stem_values,
            "prompt_texts": prompt_texts,
        }

###########################################
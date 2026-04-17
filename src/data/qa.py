import json
import logging

import datasets
import torch
from torch.utils.data import Dataset

from data.utils import add_dataset_index, load_hf_dataset, preprocess_chat_instance


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


logger = logging.getLogger("data")


class QAwithLabelDataset(Dataset):
    """
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        target_key="target_words",
        common_key="common_words",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        model_name=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.load_from_disk(
            f"data/palu/{hf_args['dataset']}/{hf_args['name']}_with_common_words_{hf_args['classifier']}"
        )
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate
        self.target_key = target_key
        self.common_key = common_key
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, target_words, common_words, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]

        assert len(prompt_msgs) == len(response_msgs)
        if isinstance(prompt_msgs, str):
            assert isinstance(response_msgs, str)
            prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

        if self.template_args["apply_chat_template"]:
            chat = []
            system_prompt = self.template_args.get("system_prompt", None)
            if system_prompt:
                chat += [{"role": "system", "content": system_prompt}]
            for prompt, response in zip(prompt_msgs, response_msgs):
                chat += [{"role": "user", "content": prompt}]
                chat += [{"role": "assistant", "content": response}]
            date_str = self.template_args.get("date_string", None)
            data_info = {"date_string": date_str} if date_str is not None else {}

            wrapped_prompt = self.tokenizer.apply_chat_template(
                chat[:-1], tokenize=False, add_generation_prompt=True, **data_info
            )
            final_response = response_msgs[-1]
            full_text = wrapped_prompt + final_response

            chat_encoded = self.tokenizer(
                full_text,
                add_special_tokens=False,  
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,  
            )
            chat_ids = chat_encoded["input_ids"]
            chat_offset_mapping = chat_encoded["offset_mapping"]

            prompt_encoded = self.tokenizer(
                wrapped_prompt,
                add_special_tokens=False,  
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=False,  
            )
            prompt_ids = prompt_encoded["input_ids"]

        else:
            wrapped_prompt = ""
            system_prompt_with_special_tokens = self.template_args.get(
                "system_prompt_with_special_tokens", None
            )
            if system_prompt_with_special_tokens:
                wrapped_prompt += system_prompt_with_special_tokens
            # add in-context examples
            n_few_shot = len(prompt_msgs) - 1  
            for i in range(n_few_shot):
                fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
                wrapped_prompt += (
                    self.template_args["user_start_tag"]
                    + fs_prompt
                    + self.template_args["user_end_tag"]
                    + self.template_args["asst_start_tag"]
                    + fs_response
                    + self.template_args["asst_end_tag"]
                )

            # add actual example
            final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
            wrapped_prompt += (
                self.template_args["user_start_tag"]
                + final_prompt
                + self.template_args["user_end_tag"]
                + self.template_args["asst_start_tag"]
            )

            chat_encoded = self.tokenizer(
                wrapped_prompt + final_response,  
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,  
            )
            chat_ids = chat_encoded["input_ids"]
            chat_offset_mapping = chat_encoded["offset_mapping"]

            prompt_encoded = self.tokenizer(
                wrapped_prompt,  
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=False,  
            )
            prompt_ids = prompt_encoded["input_ids"]

        assert (
            self.model_name is not None
        ), "model_name should be provided for QAwithLabelDataset"
        if "Llama-3" in self.model_name:
            for i in range(len(chat_offset_mapping) - 1):
                chat_offset_mapping[i] = (
                    chat_offset_mapping[i][0],
                    chat_offset_mapping[i + 1][0],
                )
            chat_offset_mapping[-1] = (
                chat_offset_mapping[-1][0],
                chat_offset_mapping[-1][0],
            )

        if chat_ids[-1] != self.tokenizer.eos_token_id:
            chat_ids += [self.tokenizer.eos_token_id]

        len_matched = len(prompt_ids)

        item = {}
        if self.predict_with_generate:
            item["input_ids"] = prompt_ids
            labels = chat_ids 
        else:
            item["input_ids"] = chat_ids
            labels = [-100] * len_matched + [-200] * (len(chat_ids) - len_matched)
            if len(prompt_ids) == len(chat_ids):
                logger.warning(
                    "Tokenization mismatch: no valid target tokens for loss computation"
                )
            len_prefix = len(wrapped_prompt)

            if target_words is not None:
                if isinstance(target_words, str):
                    target_words = json.loads(target_words)  
                for word in target_words:
                    # word = json.loads(word)
                    start, end = len_prefix + word["start"], len_prefix + word["end"]

                    for i, (encoded_start, encoded_end) in enumerate(
                        chat_offset_mapping
                    ):
                        if encoded_start < end and encoded_end > start:
                            if (
                                i >= len_matched
                            ):  
                                labels[i] = chat_ids[i]  


            if common_words is not None:
                for word in common_words:
                    word = json.loads(word)
                    start, end = len_prefix + word["start"], len_prefix + word["end"]

                    for i, (encoded_start, encoded_end) in enumerate(
                        chat_offset_mapping
                    ):
                        if encoded_start < end and encoded_end > start:
                            if (
                                i >= len_matched
                            ): 
                                if labels[i] < 0:  
                                    labels[i] = (
                                        -200
                                    )  

        item["labels"] = labels
        item["attention_mask"] = [1] * len(
            item["input_ids"]
        )  
        for attr in item:
            item[attr] = torch.tensor(item[attr])

        item_dct = {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "attention_mask": item["attention_mask"],
            "index": index,
        }

        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]

        target_words = self.data[idx][self.target_key]
        common_words = self.data[idx][self.common_key]

        if isinstance(answer, str):
            item = self._process_sample(
                question=question,
                answer=answer,
                index=index,
                target_words=target_words,
                common_words=common_words,
            )
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question,
                    answer=ans,
                    index=index,
                    target_words=target_words,
                    common_words=common_words,
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
        return return_item if self.return_original else return_item["alternate"]

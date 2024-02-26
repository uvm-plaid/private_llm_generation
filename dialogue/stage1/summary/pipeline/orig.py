import json
import os
import random
import re
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig
import torch
import spacy

import ipdb


class OrigGenerator:
    def __init__(self, args: Namespace):
        self.args = args

        self.tokenizer, self.model = self.init_tokenizer_and_model()
        self.generation_config = self.init_generation_config()

        self.spacy_model = spacy.load("en_core_web_sm")

        project_dir = os.path.dirname(os.path.dirname(__file__))

        if self.args.domain != "bio":
            with open(os.path.join(project_dir, f"data/prefix/{self.args.domain}_prefix.json"), "r") as f:
                print("FIle: ", f)
                self.prefix_resource = json.load(f)

    def init_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        if self.args.domain == "news":
            tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(self.args.device)

        elif self.args.domain == "reddit":
            tokenizer = AutoTokenizer.from_pretrained("ctrl", use_fast=False)
            tokenizer.padding_side = "left"
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            model = AutoModelForCausalLM.from_pretrained("ctrl").to(self.args.device)

        elif self.args.domain == "bio":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-large", use_fast=False)
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-large").to(self.args.device)

        elif self.args.domain == "dialog":
            tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(self.args.device)

            # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", use_fast=False)
            # tokenizer.padding_side = "left"
            # # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to(self.args.device)
            # print("MODEL: ", model)
        elif self.args.domain == "email":
            tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(self.args.device)

        else:
            raise NotImplementedError

        return tokenizer, model

    def init_generation_config(self) -> GenerationConfig:
        if self.args.domain == "news":
            bad_words = ["\n\n", "\n"]
            bad_words_ids = self.tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids + \
                self.tokenizer(bad_words, add_prefix_space=False, add_special_tokens=False).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=150, num_return_sequences=100,
                do_sample=True, top_p=0.9, temperature=1.0,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )

        elif self.args.domain == "reddit":
            bad_words_ids = [[246533]]
            generation_config = GenerationConfig(
                max_new_tokens=150, num_return_sequences=100,
                do_sample=True, top_p=0.9, temperature=1.0, no_repeat_ngram_size=3,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )

        elif self.args.domain == "bio":
            bad_words = ["<", ">", "/", "<unk>", "[", "]", "▃"]
            bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=150, num_return_sequences=100,
                do_sample=True, top_p=0.9, temperature=1.0, no_repeat_ngram_size=3,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )

        elif self.args.domain == "dialog":
            # bad_words = ["<", ">", "/", "<unk>", "[", "]", "▃"]
            bad_words = ["\n\n", "\n"]
            bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=150, num_return_sequences=2,
                do_sample=True, top_p=0.9, temperature=1.0, no_repeat_ngram_size=3,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )
        
        elif self.args.domain == "email":
            # bad_words = ["<", ">", "/", "<unk>", "[", "]", "▃"]
            bad_words = ["\n\n", "\n"]
            bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=100, num_return_sequences=100,
                do_sample=True, top_p=0.9, temperature=1.0, no_repeat_ngram_size=3,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )

        else:
            raise NotImplementedError

        return generation_config

    def generate_prefix(self) -> str:
        if self.args.domain == "news":
            city, country = random.choice(self.prefix_resource["city_country"])
            city = city.upper()
            media = random.choice(self.prefix_resource["media_list"])

            # Select template
            random_seed = random.random()

            if random_seed < 0.5:
                # Only include media name
                prefix = f"({media}) --"
            elif random_seed < 0.75:
                # Include media name and country
                prefix = f"{country} ({media}) --"
            else:
                # Include all
                prefix = f"{city}, {country} ({media}) --"

        elif self.args.domain == "reddit":
            subreddit = random.choice(self.prefix_resource["subreddit_list"])

            prefix = f"{subreddit} Text:"

        elif self.args.domain == "bio":
            topic = random.choice(["Abstract", "Introduction", "Method", "Conclusion"])

            prefix = f"{topic}:"

        elif self.args.domain == "dialog":
            # dialog_prefix = random.choice(self.prefix_resource["dialog_prefixes"])
            # prefix = f"{dialog_prefix}"
            # return random.choice(self.prefix_resource["dialog_prefixes"])

            # prefix = random.choice(self.prefix_resource["dialog_prefixes"])

            # prefix = "The following is a conversation between two old friends, John and Sarah, who unexpectedly meet at a park:"
            # initial_sentence = "John: Hey Sarah! It's been a long time. How have you been?"
            # return prefix, initial_sentence
        
            # # Select a random dialog scenario from the loaded JSON data
            scenario = random.choice(self.prefix_resource["dialog_prefixes"])
            prefix = scenario["prefix"]
            initial_sentence = scenario["initial_sentence"]
            return prefix, initial_sentence
        
        elif self.args.domain == "email":
            # dialog_prefix = random.choice(self.prefix_resource["dialog_prefixes"])
            # prefix = f"{dialog_prefix}"
            return random.choice(self.prefix_resource["email_prefixes"])
        else:
            raise NotImplementedError

        return prefix

    
    def generate_y_orig(self, prefix: Union[str, List[str]], initial_sentence=None) -> List[Tuple[str, str]]:
        if self.args.domain != "dialog":
            generation_list = self.generate_with_prefix(prefix)

            batch_pair_list = []
            for text_idx, text in enumerate(generation_list):
                if type(prefix) == str:
                    sent_list = self.postprocess_generation(prefix, text)
                else:
                    sent_list = self.postprocess_generation(prefix[text_idx // self.args.num_return_sequences], text)

                # pair sentences as x_l - y_orig
                pair_list = [(" ".join(sent_list[:i]), sent_list[i]) for i in range(1, len(sent_list))
                            if self.qualifies_as_y_orig(sent_list[i])]  # leave only the full sentences
                batch_pair_list.extend(pair_list)

            return batch_pair_list
        else:

            full_prompt = (prefix.strip() + " " + initial_sentence.strip()) if initial_sentence else prefix.strip()
            print("---"*30)
            print("PREFIX: ", prefix)
            print("INITIAL: ", initial_sentence)
            print("GENERATING OUTPUT WITH PROMPT: ", full_prompt)

            generation_list = self.generate_with_prefix(full_prompt)

            # Using regex to replace the full_prompt with initial_sentence, allowing for flexible whitespace handling
            pattern = re.compile(re.escape(full_prompt), re.IGNORECASE)
            trimmed_dialogs = [pattern.sub(initial_sentence.strip(), dialog, 1) if initial_sentence else dialog for dialog in generation_list]

            print(f"\nTRIMMED: {trimmed_dialogs}\n\n")
            dialog_list = [str(sent).strip() for sent in self.spacy_model(trimmed_dialogs[0]).sents]
            batch_pair_list = [(" ".join(dialog_list[:i]), dialog_list[i]) for i in range(1, len(dialog_list)) if self.qualifies_as_y_orig(dialog_list[i])]
            # batch_pair_list = [(" ".join(dialog_list[:i]), dialog_list[i]) for i in range(1, len(dialog_list))]
            return batch_pair_list

    #         # Dialog domain: Generate dialog based on the selected prompt
    #         full_prompt = " "+ prefix+ " " +initial_sentence
    #         print("---"*30)
    #         print("PREFIX: ", prefix)
    #         print("INITIAL: ", initial_sentence)
    #         print("GENERATING OUTPUT WITH PROMPT: ",full_prompt)

    #         generation_list = self.generate_with_prefix(prefix)
    #         # input_encoding = self.tokenizer(prefix, return_tensors="pt", padding=True).to(self.args.device)
    #         # outputs = self.model.generate(**input_encoding, **vars(self.generation_config))
    #         # generated_dialog = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #         # trimmed_dialogs = [dialog.replace(full_prompt, initial_sentence, 1) for dialog in generation_list]
    #         trimmed_dialogs = [dialog.replace(full_prompt, initial_sentence, 1) if initial_sentence else dialog for dialog in generation_list]

    #         print(f"\nTRIMMED: {trimmed_dialogs}\n\n")
    #         batch_pair_list = []
    #         for text in trimmed_dialogs:
    #             sent_list = self.postprocess_generation(full_prompt, text)

    #             # Pair sentences as x_l - y_orig, filtering by qualifies_as_y_orig
    #             pair_list = [(" ".join(sent_list[:i]), sent_list[i]) for i in range(1, len(sent_list)) if self.qualifies_as_y_orig(sent_list[i])]
    #             batch_pair_list.extend(pair_list)

    #         return batch_pair_list
                # print("\nTRIMMED: ", trimmed_dialogs)
            # batch_pair_list = []
            # for text_idx, text in enumerate(trimmed_dialogs):
            #     if type(prefix) == str:
            #         sent_list = self.postprocess_generation(prefix, text)
            #     else:
            #         sent_list = self.postprocess_generation(prefix[text_idx // self.args.num_return_sequences], text)

            #     # pair sentences as x_l - y_orig
            #     pair_list = [(" ".join(sent_list[:i]), sent_list[i]) for i in range(1, len(sent_list))
            #                 if self.qualifies_as_y_orig(sent_list[i])]  # leave only the full sentences
            #     batch_pair_list.extend(pair_list)

            # return batch_pair_list

            # dialog_list = [str(sent).strip() for sent in self.spacy_model(generated_dialog[0]).sents]

            # batch_pair_list = [(" ".join(dialog_list[:i]), dialog_list[i]) for i in range(1, len(dialog_list))]
            # return batch_pair_list


    # def generate_with_prefix(self, prefix: Union[str, List[str]]) -> List[str]:
    #     input_encoding = self.tokenizer(prefix, return_tensors="pt", padding=True).to(self.args.device)

    #     outputs = self.model.generate(
    #         **input_encoding,
    #         generation_config=self.generation_config,
    #     )

    #     outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #     return outputs_str
    
    def generate_with_prefix(self, prefix: Union[str, List[str]]) -> List[str]:
        # Reduce the memory footprint by moving inputs to GPU as needed and freeing memory after use
        input_encoding = self.tokenizer(prefix, return_tensors="pt", padding=True)
        input_encoding = {key: val.to(self.args.device) for key, val in input_encoding.items()}  # Move input encoding to GPU

        try:
            outputs = self.model.generate(
                **input_encoding,
                **vars(self.generation_config),  # Ensure this contains memory-efficient settings
            )
        finally:
            del input_encoding  # Free up memory by deleting input encoding
            torch.cuda.empty_cache()  # Optionally clear unused memory (use with caution)

        outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs_str


    def postprocess_generation(self, prefix: str, text: str) -> List[str]:
        if self.args.domain == "news":
            out = text[len(prefix):].strip()
            sent_list = [sent for sent in self.split_sentences(out)]

        elif self.args.domain == "reddit":
            out = text.strip()  # For CTRL, we leave the prefix for natural generation
            sent_list = [sent for sent in self.split_sentences(out)]

        elif self.args.domain == "bio":
            out = text[len(prefix):].strip()
            sent_list = [sent for sent in self.split_sentences(out)]

        elif self.args.domain == "dialog":
            out = text[len(prefix):].strip()
            sent_list = [sent for sent in self.split_sentences(out)]

        else:
            raise NotImplementedError

        return sent_list

    def split_sentences(self, text: str) -> List[str]:
        return [str(sent).strip() for sent in self.spacy_model(text).sents]

    def qualifies_as_y_orig(self, text: str) -> bool:
        """Given text, determine whether text qualifies as a legit y_orig"""
        if self.args.domain == "news":
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        elif self.args.domain == "reddit":
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            no_link = "http" not in text
            no_edit = len(re.findall(r'edit([\d\s]+)?:', text.lower())) == 0
            out = default and no_link and no_edit

        elif self.args.domain == "bio":
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        elif self.args.domain == "dialog":
            default = len(text) > 1 #and "\n" not in text 
            out = default 

        # elif self.args.domain == "dialog":
        #     # Basic checks for dialog text.
        #     default = len(text) > 1 and "\n" not in text

        #     # Check for conversational markers indicating a change of speaker or a dialog structure.
        #     # This can include direct address using names or pronouns, or conversational connectors like "said", "replied", "asked".
        #     conversational_markers = [" said ", " replied ", " asked ", ":", "-", "—"]  # Including common dialog punctuation like colons and dashes.
        #     contains_conversational_marker = any(marker in text for marker in conversational_markers)

        #     # Advanced check: Ensure there's an indication of a response or change in speaker.
        #     # This could be refined further with more sophisticated NLP techniques to detect direct speech, questions and answers, or alternating pronouns indicative of dialog.
        #     speaker_change_indicated = ":" in text or "-" in text or "—" in text  # Simple heuristic for speaker changes.

        #     out = default and (contains_conversational_marker or speaker_change_indicated)


        elif self.args.domain == "email":
            # For emails, ensure the text is well-structured and includes elements typical of email formatting.
            default = len(text) >= 5 and "\n" not in text  # Expecting longer sentences in emails.
            ends_with_punctuation = text[-1] in [".", "?", "!"]
            contains_salutation_or_closure = any(word in text.lower() for word in ["dear", "regards", "sincerely", "best", "thank you"])
            free_of_casual_slang = all(slang not in text.lower() for slang in ["lol", "btw", "thx", "pls"])
            out = default and ends_with_punctuation and contains_salutation_or_closure and free_of_casual_slang

        else:
            raise NotImplementedError

        return out



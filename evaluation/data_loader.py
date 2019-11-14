from torch.utils.data import Dataset
from nltk.corpus import wordnet
from transformers import BertTokenizer


class ActivityNetCaptionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maskid = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.data = self.parseData(data_file)
        
    def parseData(self, data_file):
        with open(data_file, 'r') as f:
            raw = f.readlines()
        
        data = []
        for line in raw:
            text, label = line.split('\t')
            data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataItem = self.data[idx]
        text = self.tokenizer.tokenize(dataItem[0])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.build_inputs_with_special_tokens(text))
        mask_position = indexed_tokens.index(self.maskid)
        segments_ids = self.tokenizer.create_token_type_ids_from_sequences(text)

        return (indexed_tokens, segments_ids, dataItem[1], mask_position)
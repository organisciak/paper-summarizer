import tiktoken # openai tokenizer - can estimate input size

class DocSplitter():

    def __init__(self, text, encoding_model='gpt-4'):
        ''' Split a document to chunks based on its BPE encoding'''
        self.text = text
        self.enc = tiktoken.encoding_for_model(encoding_model)
        self.fulldoc = self.enc.encode(self.text)

        break_chars = ['\n\n\n', ' \n\n', '\n\n', '   \n', '  \n', ' \n', '\n', '  ', '.'] # need to be single tokens
        self.default_break_tokens = []
        for x in break_chars:
            tok = self.enc.encode(x)
            if len(tok) == 1:
                self.default_break_tokens.append(tok[0])

    def doc_to_chunks(self, max_size=5000, window_size=50):
        '''Convert a document of tokens into more reasonably-sized chunks'''
        docs = self.split_ids(self.fulldoc, max_size=max_size, window_size=window_size)
        chunks = [self.enc.decode(doc).strip() for doc in docs]
        return chunks

    def split_ids(self, ids, max_size, break_char_tokens=None, window_size=50):
        '''Find a logical breakpoint for a chunk.
        ids: Input list of token ids.
        max_size: maximum size of a chunk
        break_char_tokens: which tokens to uses for splitting at the end
        window_size: size of window at the end of the chunk, when looking for a sensible splitting spot
        '''
        # Initialize the result list and the current sub-list
        result = []
        current_sublist = []
        if not break_char_tokens:
            break_char_tokens = self.default_break_tokens
        
        # Loop through each ID in the input list
        for i in range(len(ids)):
            # Add the current ID to the current sub-list
            current_sublist.append(ids[i])
            
            # If the current sub-list has reached the maximum size
            if len(current_sublist) == max_size:
                window_size = 50
                window_multiplier = 1
                while True:
                    window = window_size * window_multiplier
                    if any(token in break_char_tokens for token in current_sublist[-window:]):
                        # If a break character is found within the last 50 tokens, split the sub-list
                        # at the index of the last break character
                        for j in range(len(current_sublist) - window, len(current_sublist)):
                            if j < 0:
                                break
                            if current_sublist[j] in break_char_tokens:
                                result.append(current_sublist[:j+1])
                                current_sublist = current_sublist[j+1:]
                                break
                    window_multiplier += 1
                    if window_multiplier > 3:
                        result.append(current_sublist[:max_size+1])
                        current_sublist = current_sublist[max_size+1:]
                        break
        
        # Add the final sub-list to the result list
        if len(current_sublist) > 0:
            result.append(current_sublist)
        
        return result
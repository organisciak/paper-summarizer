import tiktoken # openai tokenizer - can estimate input size
import re
class DocSplitter():

    def __init__(self, text, encoding_model='gpt-4', drop_refs=True):
        ''' Split a document to chunks based on its BPE encoding'''
        self.text = text
        self.enc = tiktoken.encoding_for_model(encoding_model)

        cleaned = self.clean_doc(text)
        self.fulldoc = self.enc.encode(cleaned)
        if drop_refs:
            split_at = self.find_doi_split()
            self.fulldoc = self.fulldoc[:split_at]

        # splitting tokens, in order of preference
        break_chars = ['\n\n\n', ' \n\n', '\n\n'] + [' '*x+'\n' for x in range(5)] + ['  ', '.'] # need to be single tokens
        self.default_break_tokens = []
        for x in break_chars:
            tok = self.enc.encode(x)
            if len(tok) == 1:
                self.default_break_tokens.append(tok[0])

    def clean_doc(self, text):
        # remove whitespace at start of lines
        cleaned = re.sub('^\s+', '', text, flags=re.MULTILINE)
        return cleaned
    
    def doc_to_chunks(self, max_size=5000, window_size=50):
        '''Convert a document of tokens into more reasonably-sized chunks'''
        docs = self.split_ids(self.fulldoc, max_size=max_size, window_size=window_size)
        chunks = [self.enc.decode(doc).strip() for doc in docs]
        return chunks
    
    def find_doi_split(self, window_size=1000):
        '''move a cursor through the doc and look for when the lookback has notably less
        'doi' tokens than lookahead. Returns an index for splitting.'''
        target_text = 'doi'
        tok = self.enc.encode(target_text)
        assert len(tok) == 1, "Has to target a 1 token word"
        tok = tok[0]

        best_diff, best_diff_at = 0, -1
        n = len(self.fulldoc)
        start_at = 2*n//3 # start in last third, to avoid failures being too bad
        for cursor in range(start_at, n): # start n
            before = self.fulldoc[cursor-window_size:cursor]
            after = self.fulldoc[cursor:cursor+window_size]
            
            before_doi_n = len([x for x in before if x == tok])
            after_doi_n = len([x for x in after if x == tok])
            diff = after_doi_n - before_doi_n
            if diff >= best_diff: # deliberately >=, to get the latest option
                best_diff = diff
                best_diff_at = cursor
        return best_diff_at

    def split_ids(self, ids, max_size, break_char_tokens=None, window_size=50):
        if break_char_tokens is None:
            break_char_tokens = self.default_break_tokens

        all_chunks = []
        remaining_ids = ids
        
        def get_last_index(lst, value):
            for i, x in enumerate(lst[::-1]):
                if x == value:
                    return -1-i
            return None

        def get_final_break(doc):
            # search tail for a breaking character, before widening the window
            break_char_inds = [get_last_index(doc, tok) for tok in break_char_tokens]
            break_char_inds = [x for x in break_char_inds if x]
            if not len(break_char_inds):
                # stop trying fancy splits and just return the chunk
                return len(doc)+1
            
            # expand the window three times, but prioritize smaller windows over earlier break_chars
            for window_multiplier in range(3):
                window = window_multiplier * window_size
                for x in break_char_inds:
                    if abs(x) <= window:
                        return x+1
                    
        while True:
            candidate_chunk, remaining_ids = remaining_ids[:max_size], remaining_ids[max_size:]

            if len(remaining_ids) == 0:
                all_chunks.append(candidate_chunk)
                break

            final_break = get_final_break(candidate_chunk)
            if not final_break:
                final_break = len(candidate_chunk)
            candidate_chunk, remainder = candidate_chunk[:final_break], candidate_chunk[final_break:]
            if len(remainder):
                remaining_ids = remainder + remaining_ids
            all_chunks.append(candidate_chunk)

        return all_chunks
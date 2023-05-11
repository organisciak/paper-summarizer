import PyPDF2
from tqdm.auto import tqdm
import openai
import re
from yaml_sync import YamlCache
from pathlib import Path
import hashlib
from .utils import DocSplitter
from IPython.display import display_markdown, Markdown

class PaperSummarizer:
    def __init__(self, text, cache=True, cache_location=None, model='gpt-3.5-turbo', drop_refs=True):
        self.cache_location = cache_location
        self.model = model
        self.text = text

        assert not (cache and cache_location is None), "Set a location for yaml file with cache_location if cache is True"
        
        splitter = DocSplitter(self.text, encoding_model=model, drop_refs=drop_refs)
        chunksize = 2*4097//3
        if model == 'gpt-4':
            chunksize = 2*8192//3
        self.chunks = splitter.doc_to_chunks(max_size=chunksize)
        
        if cache:
            self.cache = YamlCache(self.cache_location, mode='rw',
                                   order=['model', 'full_summary', 'questions',
                                          'full_points', 'chunk_summaries', 'chunk_keypoints'])
        else:
            self.cache = {}
        self.cache['model'] = model

    def question(self, question, model='default', temperature=0, force=False, md=False):
        ''' Once the full summary and keypoints are created, you can ask ad hoc questions.'''
        if model == 'default':
            model = self.model

        if ('questions' in self.cache) and (question in self.cache['questions']) and not force:
            cached_response = self.cache['questions'][question]
            if md:
                cached_response = Markdown(cached_response)
            return cached_response
        
        summary = self.full_summary()
        pts = self.full_points()

        big_summary = f'''The follow is a summary of a research paper and it's key points.

# SUMMARY
{summary}

# KEY POINTS
{pts}

-------------

Are you ready to answer questions about the paper?
'''

        big_summary ={"role":"user", "content": big_summary}
        bot_affirm = {"role":"assistant", "content": "Yes, I am ready to answer questions about the paper."}

        result = openai.ChatCompletion.create(
                    model=model,
                    messages=[big_summary, bot_affirm,
                    {"role": "user", "content": question}
                    ],
                    temperature = temperature,
            )
        
        if 'questions' not in self.cache:
            self.cache['questions'] = {}
        qs = self.cache['questions']
        qs[question] = result.choices[0].message.content
        self.cache['questions'] = qs

        if md:
            return Markdown(qs[question])
        else:
            return qs[question]
    
    
    def summarize(self, md=False):
        ''' Print full summaries'''
        summary = self.full_summary()
        pts = "\n- " + "\n- ".join(self.full_points())

        out = f'## Summary\n{summary}\n## Key Points\n{pts}'
        if md:
            display_markdown(Markdown(out))
        else:
            print(out)

    def summarize_all_chunks_raw(self, force=False):
        ''' Iterate through chunks and get summaries and key points for each one '''
        if ('chunk_summaries' in self.cache) and ('chunk_keypoints' in self.cache) and not force:
            return self.cache['chunk_summaries'], self.cache['chunk_keypoints']
        
        all_summaries, all_pts = [], []
        for chunk in tqdm(self.chunks):
            summary, pts = self.summarize_chunk(chunk)
            all_summaries.append(summary)
            all_pts.append(pts)

        self.cache['chunk_summaries'] = all_summaries
        self.cache['chunk_keypoints'] = all_pts

        return self.cache['chunk_summaries'], self.cache['chunk_keypoints']

    def summarize_chunk(self, chunk, model='default', temperature=0, raw_response=False):
        ''' Summarize a chunk of a document, into a summary and outline ''' 
        if model == 'default':
            model = self.model

        msg1 = f'''Extract the key information from the following chunk of a research paper.
Material that may be important is: justification, background information, methods used, experimental results - with quants if available, discussion, important considerations, and broad impact.

Format the output in markdown, with two headings: 'SUMMARY' and 'OUTLINE'.

SUMMARY should be a summary of 600 words or less.

OUTLINE should be a point-by-point outline of this chunk of the article. Exclude copyright information and paratextual materil.

Here is the paper to summarize:

PAPER
-------

{chunk}
'''

        result = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": msg1.strip()}],
            temperature = temperature,
        )
        if raw_response:
            return result
        result_txt = result.choices[0].message.content
        summary, keypts = result_txt.split('OUTLINE')
        summary = summary.split('\n', 1)[1].strip()
        keypts = keypts.split('\n', 1)[1].strip()
        keypts = self._parse_md_list(keypts)
        return summary, keypts
  
    def full_summary(self, model='default', temperature=0, force=False):
        ''' Combine the chunk summaries into one full paper summary'''
        if model == 'default':
            model = self.model

        if 'full_summary' in self.cache and not force:
            return self.cache['full_summary']

        chunk_summaries, _ = self.summarize_all_chunks_raw()

        msg = '''The follow is a summary of different chunks of the same research paper. Combine them into one overall summary. Note the problem, approach, justification, methods employed, experimental results, broader impact, and other pertinent information.\n\n'''
        for i, chunk_summary in enumerate(chunk_summaries):
            msg += f"# CHUNK {i}\n{chunk_summary}\n\n"
        
        result = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": msg}],
            temperature = temperature,
        )
        self.cache['full_summary'] = result.choices[0].message.content
        return self.cache['full_summary']
    
    def _parse_md_list(self, list_txt):
        l = [x.strip() for x in list_txt.strip('-').split('\n-')]
        return l

    def full_points(self, model='default', temperature=0, force=False):
        ''' Combine the by-chunk key points into one set'''
        if model == 'default':
            model = self.model

        if 'full_points' in self.cache and not force:
            return self.cache['full_points']
        
        _, chunk_keypoints = self.summarize_all_chunks_raw()

        chunk_keypoints = [pt for chunk in chunk_keypoints for pt in chunk]
        in_pts = "- " + "\n- ".join(chunk_keypoints)

        msg = f'''The following is a summary of key points extracted from different chunks of the same paper.

        Consolidate the points for readability and to reduce redundancy, and present with the most important information first. Return the information as bullet points.

        Include information such as problem and approach, justification, methods, experimental results - including quants when relevant, significance and impact, and future ramifications. Don't include points unrelated to the research, such as copyright statemements and calls for reviewers.

        Format the results as Markdown bullet points.

        # INPUT POINTS

        {in_pts}
        '''

        result = openai.ChatCompletion.create(
                model=model,
                messages=[
                {"role": "user", "content": msg}
                ],
                temperature = temperature,
        )
        result_txt = result.choices[0].message.content
        keypts_txt = re.sub('^#.*', '', result_txt).strip()
        self.cache['full_points'] = self._parse_md_list(keypts_txt)
        return self.cache['full_points']

class PDFPaperSummarizer(PaperSummarizer):
    def __init__(self, pdf_path, cache_location='match_file', *args, **kwargs):
        self.pdf_path = pdf_path
        
        text = self.extract_text_from_pdf()

        if cache_location == 'match_file':
            cache_location = f"{pdf_path.rsplit('.', 1)[0]}.yaml"
        elif cache_location == 'hash':
            # saves to the same directory, but append the hash of the text to the name
            p = Path(pdf_path)
            md5_hasher = hashlib.md5()
            md5_hasher.update(self.text.encode('utf-8'))
            hash = md5_hasher.hexdigest()[:6]
            cache_location = p.parent / f"{p.stem}_{hash}.yaml"

        super().__init__(text, cache_location=cache_location, *args, **kwargs)

    def extract_text_from_pdf(self):
        with open(self.pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            extracted_text = ""
            for page_num, page in enumerate(reader.pages):
                extracted_text += page.extract_text()

        return extracted_text
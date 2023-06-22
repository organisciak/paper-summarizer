import PyPDF2
from tqdm.auto import tqdm
import openai
import re
from yaml_sync import YamlCache
from pathlib import Path
import hashlib
from .utils import DocSplitter
import numpy as np
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
        elif model == 'gpt-3.5-turbo-16k':
            chunksize = 2 * 16384//3
        elif model == 'gpt-4-32k':
            chunksize = 2*32768//3
        
        self.chunks = splitter.doc_to_chunks(max_size=chunksize)
        
        if cache:
            self.cache = YamlCache(self.cache_location, mode='rw',
                                   order=['model', 'full_summary', 'questions',
                                          'full_points', 'chunk_outlines'])
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
    
    
    def summarize(self, md=False, protocol=0, force=False):
        ''' Print full summaries'''
        summary = self.full_summary(protocol=protocol, force=force)
        pts = "\n- " + "\n- ".join(self.full_points(protocol=protocol))

        out = f'## Summary\n{summary}\n## Key Points\n{pts}'
        if md:
            display_markdown(Markdown(out))
        else:
            print(out)

    def outline_all_chunks(self, force=False, protocol=0, debug=False):
        ''' Iterate through chunks and get summaries and key points for each one.
        
        Protocol is an integer referring to a specific prompt style, and is passed down to outline_chunk.
        
        '''
        if ('chunk_outlines' in self.cache) and (len(self.cache['chunk_outlines']) == len(self.chunks)) and not force:
            return self.cache['chunk_outlines']
        
        pbar = tqdm(total=len(self.chunks), desc="Paper chunk summarizing progress", position=1, leave=False)
        all_pts = []
        for i, chunk in enumerate(self.chunks):
            if ('chunk_outlines' in self.cache) and (i < len(self.cache['chunk_outlines'])) and not force:
                    # resume previously interrupted process that already has *some* outlines
                    pbar.update(1)
                    continue
            pts = self.outline_chunk(chunk, protocol=protocol, debug=debug)
            all_pts.append(pts)
            # save intermediate progress
            self.cache['chunk_outlines'] = all_pts
            pbar.update(1)

        self.cache['chunk_outlines'] = all_pts

        if len(self.chunks) == 1:
            print("Only one chunk - saving as full description")
            self.cache['full_points'] = pts

        return self.cache['chunk_outlines']

    def outline_chunk(self, chunk, model='default', temperature=0, raw_response=False, protocol=0, debug=False):
        ''' Summarize a chunk of a document, into a summary and outline.

        The protocol is an integer referring to a specific prompt style.
            0: GPT is asked to outline each chunk.
            1: GPT is asked to rewrite the chunk at 25% of the length and point form. This may tease out longer outlines, which is useful at this stage.
            2: This asks for each paragraph to be described. It tries to tease out as much detail as possible, while compressing the token count. This may get too long.
        ''' 
        if model == 'default':
            model = self.model

        max_len = min(np.round(100/len(self.chunks), 0).astype(int), 20) # max length of summary is 100%/len(chunks) or 20% of the total length of the paper
        
        protocols = [
        f'''Outline the following chunk of a research paper, point-by-point, with as much details as is needed to capture all arguments.
Material that may be important is: justification, background information, methods used, experimental results - with quants if available, discussion, important considerations, and broad impact.

Format the output in Markdown, under the heading 'OUTLINE'.

OUTLINE should be a finely detailed point-by-point outline of this chunk of the article. Exclude copyright information and paratextual material.

Here is the paper to summarize:

PAPER
-------

{chunk}
''',

f'''Summarize every paragraph in the following chunk of a research paper. Be as detailed as possible. The summary should be no more than {max_len}% of the length (i.e. trim it by {1-max_len}%). Capture all arguments.
Material that may be important is: justification, background information, methods used, experimental results - with quants if available, discussion, important considerations, and broad impact. Exclude copyright information and paratextual material.

Format the output in Markdown, under the heading 'OUTLINE'. Include paragraph number references for each point.

Here is the document to summarize:

PAPER
-------

{chunk}
''',

f'''Paragraph by paragraph, outline every paragraph in this paper in 5000 words, with the paragraph number cited.

Format the output in Markdown, under the heading 'OUTLINE'. Include paragraph number references for each point.

Here is the document to summarize:

PAPER
-------

{chunk}
'''

        ]
        msg = protocols[protocol]
        if debug:
            print(msg[:2500])

        result = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": msg.strip()}],
            temperature = temperature,
        )
        if raw_response:
            return result
        result_txt = result.choices[0].message.content
        _, pts = result_txt.split('OUTLINE')
        pts = pts.split('\n', 1)[1].strip()
        pts = self._parse_md_list(pts)
        return pts
  
    def full_summary(self, model='default', temperature=0, force=False, protocol=0):
        ''' Combine the chunk summaries into one full paper summary'''
        pbar = tqdm(total=2, desc="Preparing full summary", position=0, leave=True)
        if model == 'default':
            model = self.model

        if 'full_summary' in self.cache and not force:
            pbar.update(2)
            return self.cache['full_summary']
        if force:
            print("Forcing a rewrite of the chunk condensation - to force the re-outlining of chunks, run outline_all_chunks(force=True) first.")

        pbar.set_description('Outlining paper chunks')
        all_outlines = self.outline_all_chunks(protocol=protocol)
        pbar.update(1)
        pbar.set_description('Combining outline notes to a full summary')

        protocols = [
            '''The following is a detailed outline of multiple chunks of the same research paper. Combine them into one overall summary as fully as possible. Note the problem, approach, justification, methods employed, experimental results, broader impact, and other pertinent information.\n\n''',
            '''The following is a detailed outline of a same research paper. Rewrite to capture the most important points in 1000 words.\n\n'''
        ]
        msg = protocols[protocol]
        for i, chunk_outline in enumerate(all_outlines):
            msg += f"# CHUNK {i}\n{chunk_outline}\n\n"
        
        result = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": msg}],
            temperature = temperature,
        )
        self.cache['full_summary'] = result.choices[0].message.content
        pbar.update(1)
        return self.cache['full_summary']
    
    def _parse_md_list(self, list_txt):
        l = [x.strip() for x in list_txt.strip('-').split('\n-')]
        return l

    def full_points(self, model='default', temperature=0, force=False, protocol=0):
        ''' Combine the by-chunk key points into one set'''
        if model == 'default':
            model = self.model

        if 'full_points' in self.cache and not force:
            return self.cache['full_points']
        
        chunk_outlines = self.outline_all_chunks(protocol=protocol)

        chunk_outlines = [pt for chunk in chunk_outlines for pt in chunk]
        in_pts = "- " + "\n- ".join(chunk_outlines)

        msg = f'''The following is a details outline of key points from multiple chunks of the same research paper.

        Combine all the points as fully and completely as possible, organized sensibly, avoiding redundancy but including as much non-redundant detail as possible. Present the most important information first. Return the information as bullet points.

        Include information such as problem and approach, justification, methods, experimental results - including quants when relevant, significance and impact, and future ramifications. Don't include points unrelated to the research, such as copyright statemements and calls for reviewers.

        Format the results as Markdown bullet points.

        # INPUT OUTLINE

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
    

class DocxPaperSummarizer(PaperSummarizer):
    def __init__(self, doc_path, cache_location='match_file', *args, **kwargs):
        self.doc_path = doc_path
        
        text = self.extract_text_from_docx()

        if cache_location == 'match_file':
            cache_location = f"{doc_path.rsplit('.', 1)[0]}.yaml"
        elif cache_location == 'hash':
            # saves to the same directory, but append the hash of the text to the name
            p = Path(doc_path)
            md5_hasher = hashlib.md5()
            md5_hasher.update(self.text.encode('utf-8'))
            hash = md5_hasher.hexdigest()[:6]
            cache_location = p.parent / f"{p.stem}_{hash}.yaml"

        super().__init__(text, cache_location=cache_location, *args, **kwargs)

    def extract_text_from_docx(self):
        from docx import Document
        document = Document(self.doc_path)

        text = ""
        for paragraph in document.paragraphs:
            formatted_paragraph = self.markdown_formatting(paragraph)

            # Write the formatted text to the file
            text += formatted_paragraph.strip() + "\n\n"
        return text.strip()
    
    def markdown_formatting(self, paragraph):
        """
        This function receives a paragraph object and iterates over its runs,
        formatting bold and italic text with Markdown and returning the resulting string.
        """
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

        bold, bolditalic, italic, regular = 0, 0, 0, 0
        text = ""
        # count how much of the text is a certain style (for making heading inferences)
        for i, run in enumerate(paragraph.runs):
            text += run.text
            if run.text.strip() == '':
                continue
            if run.bold:
                bold += len(run.text)
            elif run.bold and run.italic:
                bolditalic += len(run.text)
            elif run.italic:
                italic += len(run.text)
            else:
                regular += len(run.text)

        total_len = len(text.strip())
        
        if paragraph.style.name == "Normal":
            # make guesses for a 'Normal formatted' paragraph
            if total_len <= bold:
                if paragraph.alignment == WD_PARAGRAPH_ALIGNMENT.CENTER:
                    # Heading: Bold and Centered
                    return f"# {text.strip()}"
                else:
                    # Subheading: Bold
                    return f"## {text.strip()}"
            elif total_len <= bolditalic:
                # Subsubheading: Bold and Italic
                return f"### {text.strip()}"
            elif total_len <= italic:
                # Subsubsubheading: Italic
                return f"#### {text.strip()}"
            else:
                # Apply formatting to individual runs
                formatted_text = ""
                for run in paragraph.runs:
                    end_spaces = run.text[len(run.text.rstrip()):]
                    start_spaces = run.text[:len(run.text) - len(run.text.lstrip())]
                    if run.text.strip() == '':
                        formatted_text += run.text
                    elif run.bold and run.italic:
                        # Both bold and italic
                        formatted_text += f"{start_spaces}***{run.text.strip()}***{end_spaces}"
                    elif run.bold:
                        # Only bold
                        formatted_text += f"{start_spaces}**{run.text.strip()}**{end_spaces}"
                    elif run.italic:
                        # Only italic
                        formatted_text += f"{start_spaces}_{run.text.strip()}_{end_spaces}"
                    else:
                        # Regular text
                        formatted_text += run.text
                # fix any formatting within word
                formatted_text = formatted_text.replace('****', '')
                formatted_text = re.sub('\*\* +\*\*', r' ', formatted_text)
                #formatted_text = re.sub('(\W)([\*\_]{1,2})', r'\1 \2', formatted_text)
                #formatted_text = re.sub('(\w)[\*\_]{1,2}(\w)', r'\1\2', formatted_text)
                return formatted_text

        elif 'Heading' in paragraph.style.name:
            level = paragraph.style.name.split(' ')[1]  # Assume style is like 'Heading 1'
            return f"{'#' * int(level)} {text.strip()}"

        else:
            return text
# Paper Summarizer

Paper Summarizer is a Python library for extracting and summarizing content from research papers. It utilizes OpenAI's language models to generate comprehensive summaries and key points from PDF documents.

## Installation

```bash
pip install paper_summarizer
```

## Basic Usage

This library uses the OpenAI API. You need to specify your API key. Here are two ways:

```python
import openai

# if key in file
openai.api_key_path = 'path/to/key'

# if key in environmental variable
openai.api_key = os.getenv("OPENAI_API_KEY")
```

`summarize()` generates a long summary, which is used for subsequent shorter summaries and questions asking.

```python
from paper_summarizer import PaperSummarizer

# Initialize the summarizer with a PDF path
summarizer = PDFPaperSummarizer("path/to/your/document.pdf", model='gpt-3.5-turbo')

# Generate a summary
summary = summarizer.summarize()
```

Generating a summary the first time may take a few minutes. First, it chunks the full document into chunks that are up to two-thirds of the max input size of the model (for ChatGPT, or `gpt-3.5-turbo`, that's 4,097, for `gpt-4` that's 8,192). Each chunk is outlined individually, then they're combined into a single paper summary and outline of key points.


`question()` can pose ad-hoc questions and requests. The `md` argument returns a Ipython `Markdown` object which renders nicely in notebooks.

```python
summarizer.question("Rewrite this paper as a series of haikus.", md=True)
```

## Features
- Extract text from PDF documents
- Split large documents into manageable chunks for summarization
- Cache summaries as yaml to speed up repeated queries while offering a human-readable view of summaries and outputs.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. This started as quick proof-of-concept, so there's a lot to do - at the same time, I hope not to over-engineer this thing.

## Contact

Peter Organisciak <https://porg.dev>
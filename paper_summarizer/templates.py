# flake8: noqa: E122
TEMPLATES = {}

TEMPLATES['big_summary'] = '''The following is a summary of a {doctype} and it's key points.

# SUMMARY
{summary}

# KEY POINTS
{pts}

-------------

Are you ready to answer questions about the {doctype}?
'''

TEMPLATES['chunk_outline'] = [
'''Outline the following chunk of a {doctype}, point-by-point, with as much details as is needed to capture all arguments.
Material that may be important is: justification, background information, methods used, experimental results - with quants if available, discussion, important considerations, and broad impact.

Format the output in Markdown, under the heading 'OUTLINE'.

OUTLINE should be a finely detailed point-by-point outline of this chunk of the article. Exclude copyright information and paratextual material.

Here is the document to summarize:

{doctype}
-------

{chunk}
''',

'''Summarize every paragraph in the following chunk of a {doctype}. Be as detailed as possible. The summary should be no more than {max_len}% of the length (i.e. trim it by {1-max_len}%). Capture all arguments.
Material that may be important is: justification, background information, methods used, experimental results - with quants if available, discussion, important considerations, and broad impact. Exclude copyright information and paratextual material.

Format the output in Markdown, under the heading 'OUTLINE'. Include paragraph number references for each point.

Here is the document to summarize:

{doctype}
-------

{chunk}
''',

'''Paragraph by paragraph, outline every paragraph in this {doctype}, totaling 5000 words, with the paragraph number cited.

Format the output in Markdown, under the heading 'OUTLINE'. Include paragraph number references for each point.

Here is the document to summarize:

{doctype}
-------

{chunk}
''',

'''Rewrite the following document to half the length.

Format the output in Markdown, under the heading 'OUTLINE'. Include paragraph number references for each point.

Here is the document to summarize:

{doctype}
-------

{chunk}
'''
]
# fakeplasticgrass
## Detecting Astroturfing by Plastic Waste Non-Profits (completed for WRDS 150B, winter 2020)

The first step to this analysis was website sourcing. The keywords “plastic pollution organisations” were entered into Google Canada. To focus on high-impact organisations, only the first five pages of search results were examined. In order to qualify, organisations had to feature plastic waste reduction as their primary target, and their websites had to be in English with an About, Mission, and/or Vision page. These pages were chosen since they communicate an organisation’s strategic goals to a general audience. In addition, organisations had to declare their primary funding sources so that they could be labelled as grassroots (mostly individual donations) or astroturfing (mostly large corporations). From search results, 26 organisations met these criteria. Four were excluded because of limited English-language content and eight because of for-profit business models. A total of 14 organisations (9 grassroots, 5 astroturf) met eligibility criteria. These sources are listed in Table 1.

All text content on the About, Mission, and/or Vision pages of a given site was scraped into a file. Two Python scripts were written to process these files. The first script (tokenize) runs files through three readability analyses (ARI, Flesch-Kincaid, and Smog) encoded within the py-readability-metrics library. These standardised formulae calculate the approximate grade level required for understanding a given text sample. The script then analyses the files using Python’s Natural Language Toolkit. Because this project aims to identify themes from vocabulary, it uses a bag-of-words approach, where word order is unimportant: this is a common and effective strategy for identifying and comparing themes across documents.

Once text had been collected from these sources, the data were cleaned: each file was lemmatized, a process which simplifies words to their stems (i.e. “made” to “make”), punctuation was stripped, and “stop words” (such as prepositions) were removed. This ensured that only the most meaningful words underwent analysis, and that variations of one word were aggregated to reflect all usages. Basic metrics were then analysed. Internal comparisons between cleaned texts were performed within the two categories (grassroots and astroturf) to find words that appeared frequently across multiple texts, and the two resulting lists were compared to identify common elements. These results were compiled into a Venn diagram. The words were also qualitatively coded into subcategories to derive overarching themes.

The second Python script (lda-scikit-loop) performs latent Dirichlet allocation. Two separate models were written, using the gensim and scikit-learn libraries respectively. The scikit-learn model (Appendix II) was selected due to lower overlap between topics. Once two topics had been extracted from the corpus, each document underwent individual analysis to determine which topic fit it best. The script was looped several thousand times, deriving slightly different topics each iteration, until at least 80% of grassroots organisations were assigned to one topic and at least 80% of astroturfing organisations were assigned to the other topic.

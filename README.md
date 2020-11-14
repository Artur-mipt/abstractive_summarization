# topical_summarization

## LexRank

[Biased LexRank](https://duc.nist.gov/pubs/2006papers/33.pdf) algorithm implementation.

The set of sentences in a document cluster is represented as a graph, where nodes
are sentences and links between the nodes are induced by a similarity relation between the sentences. Then
the sentences are ranked according to a random walk model defined in terms
of both the inter-sentence similarities and the similarities of the sentences to
the topic description.

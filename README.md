### DNA---introns-exons-
##SUMMARY
>The study of DNA sequencing enables the development of new methods to diagnose diseases and the formulation of new drugs. Protein-coding genes constitute the functional part of DNA, where the basic biological instructions for gene expression (protein production) are located. Advances in the automation of DNA sequencing have made it possible to create large amounts of data, generating a high demand for more efficient analysis methods. Computational techniques such as machine learning have been widely used, but the general performance of these methods is still not considered satisfactory. In this work we use an aggregate of machine learning algorithms to identify the nucleotides that best identify the subsequence in order to identify introns and exons regions.
#1. INTRODUCTION
> Advances in DNA sequencing technology and automation have made it possible to generate vast amounts of DNA sequence data. This large data growth generated a significant demand for more efficient analysis methods, such as the use of computational techniques. The precise location of introns and exons junctions and the understanding of the gene structure in these large databases is one of the main topics of study in the field of bioinformatics, as it allows the development of new methods to diagnose diseases and the formulation of new drugs.
>Several computational methods have been proposed in the last decade for the identification of eukaryotic genes. In particular, approaches that use machine learning techniques have been explored to identify patterns in DNA sequences. However, the general performance of these algorithms for gene prediction is still not considered satisfactory.
>In this context, in this work, our objective is to apply voting algorithms to improve the performance of introns and exons segment predictions. In this way, we develop and evaluate prediction models using the combined machine learning algorithms for the automatic identification of exon-intron and intron-exon junction points in genes.
#2 THEORETICAL FOUNDATION
> Junction Points: The human genome contains approximately 23,500 nucleotide-encoding genes [3]. The process of decoding proteins from the information contained in genes is called gene expression. Genes are composed alternately of segments called introns and exons. Exons are the regions of the gene that will be translated into proteins; the introns are the non-coding regions. In Figure 1 a schematic representation of the protein decoding process is presented. The understanding of gene expression can only be achieved from the correct identification of the junction areas in the sequences corresponding to the genes and the type of junction (exon-intron or intron-exon) [5].
>Machine Learning: the acquisition of knowledge automatically through machine learning (MA) algorithms is based on inductive inference, in which knowledge can be derived from previously known ones. In this way, a hypothesis is induced from a set of examples (sample) that allow characterizing the domain (population) to be treated. According to the type of concept used to induce a certain hypothesis, AM algorithms can be divided into paradigms: Symbolic paradigm: they carry out the process of learning a concept using symbolic representations through the analysis of examples and counter-examples. Usually represented in the form of a logical expression, such as the decision tree (DT) or semantic network. Paradigm based on examples: save the examples and use similarity measures to identify the cases most similar to the example to be analyzed. For example k-nearest neighbors (KNN). Statistical paradigm: use statistical models to find an approximation of the induced concept. Among these algorithms, support vector machines (SVM) and Bayesian learning (GNB) stand out. Connectionist paradigm: they use mathematical constructions inspired by neuronal connections in the human nervous system. Examples of this are artificial neural networks (MLP) [4].
>Ensemble methods: instead of looking for the best hypothesis to explain the data, these algorithms build a set of hypotheses (set of AM), then make these hypotheses “vote” in some way to predict the class of new data. Experimental evidence shows that ensemble methods are generally much more accurate than either hypothesis alone.
#3 METHODOLOGY
> The method proposed in this work can be structured in two main steps: a) Data transformation; b) Separation of the database together with the construction of AM and classification models. In the first step (a), initially the DNA sequences were converted to numbers, such as: A to 0, C to 1, G to 2, T to 3. For example, the CCAGCT sequence would become 110213. In the second step ( b) The data are separated into 3 parts, 80% of the sequences of each class are separated for the training of the algorithms, 10% for the selection of vote weights for each algorithm, 10% for the tests (technique based on crossvalidation), the algorithms are trained from the training set, they predict the classes of the weight selection set, the weights are defined by the number of hits in the predictions of the class divided by the total number of guesses of this class, that is, each model will have a weight for each class (intron-exon, exon-intron, none), then each algorithm makes the predictions from the test set, and the algorithm created for the vote makes the prediction based on the class that has the most points, the point being counted from the sum of 1*weight of the prediction of each model. models of X AM algorithms were built.
> As mentioned, in this work we seek to apply machine learning (AM) techniques in DNA sequences for the identification of intron-exon (i-e) and exon-intron (e-i) junctions. This approach is due to the large amounts of data and the great efficiency of these AM methods. For our tests, we used the Molecular Biology (Splice-junction Gene Sequences) Data-Set database, which contains 3190 DNA subsequences with 60 nucleotides each , classified according to type: i-e, e-i or none (N). In this dataset, all examples are taken from Genbank 64.1, and are all part of primate genes. For data manipulation and application of algorithms, we used the Python language because it is practical and commonly used by the scientific community. For machine learning algorithms we use the scikit-learn library (http://scikit-learn.org). All algorithms were used in the default configuration.
#5 CONCLUSIONS
> Based on the results presented in this work, it was observed that the application of ensemble methods to identify the most important nucleotides made it possible to create a classifier with a higher precision than the individual algorithms. Additionally, it is interesting to note that the voting classifier highlighted the qualities of each algorithm in the prediction of each class, by connecting the prediction with the performance obtained by the algorithm when predicting that class.

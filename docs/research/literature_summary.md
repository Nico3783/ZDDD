# Literature Summary

## Key References

### Garcia-Teodoro, Diaz-Verdejo, et al. (2009)

**Anomaly-based network intrusion detection: Techniques, systems and challenges**

Foundational survey establishing anomaly-based detection as an alternative to signature-based IDS. The authors argue that anomaly detection can identify novel attacks by modeling normal behavior and flagging deviations. Key contributions include taxonomy of detection techniques (statistical, knowledge-based, machine learning) and identification of challenges: concept drift, evasion, and high false positive rates. This work motivates the use of Isolation Forest for zero-day detection.

### Sommer and Paxson (2010)

**Outside the closed world: On using machine learning for network intrusion detection**

Seminal paper examining ML application to intrusion detection. Key finding: ML-based IDS must handle class imbalance (benign vastly outnumbers attacks) and concept drift (attack patterns evolve). Recommends flow-based features over raw packets and emphasizes the importance of feature engineering. Directly relevant to the 30-feature selection approach used in this project.

### Buczak and Guven (2016)

**A survey of data mining and machine learning methods for cyber security intrusion detection**

Comprehensive survey of ML methods applied to IDS. Compares supervised (SVM, Random Forest, decision trees) and unsupervised (clustering, autoencoders) approaches. Finding: Random Forest consistently achieves top performance for attack classification with interpretable feature importance. Supports the choice of RF as the classification model.

### Mirsky, Doitshman, Elovici, and Shabtai (2018)

**Kitsune: An ensemble of autoencoders for online network intrusion detection**

Introduces Kitsune, an online anomaly detection system using ensemble autoencoders. Demonstrates real-time network intrusion detection without labeled training data. While using deep learning rather than Isolation Forest, the work validates the unsupervised anomaly detection paradigm for zero-day detection and provides benchmark comparisons.

### Chalapathy and Chawla (2019)

**Deep learning for anomaly detection: A survey**

Survey covering deep learning approaches for anomaly detection including autoencoders, GANs, and RNNs. Key insight: traditional ML methods (Isolation Forest, One-Class SVM) remain competitive for tabular network flow data, while deep learning excels at complex patterns in unstructured data. Supports the decision to use Isolation Forest over deep learning for this project.

## Supporting References

### Ahmad and Frigui (2019)

**Adaptive thresholding for anomaly detection in network traffic**

Proposes adaptive threshold selection methods that adjust based on traffic characteristics. Relevant to the multi-tier threshold strategy (critical/high/medium/low) implemented in the detection engine.

### Liu, Ting, and Zhou (2008)

**Isolation forest**

Original Isolation Forest paper. Establishes the isolation-based anomaly detection paradigm. Key finding: isolation is more efficient than distance-based or density-based methods for high-dimensional data. The algorithm's efficiency makes it suitable for real-time detection.

### Breiman (2001)

**Random forests**

Original Random Forest paper. Establishes the ensemble decision tree method with bootstrap aggregation and feature randomness. Key properties: handles high-dimensional data, robust to overfitting, provides feature importance rankings. Foundation for the classification model in this project.

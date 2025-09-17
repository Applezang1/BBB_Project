# Paper 3: LightBBB: computational prediction model of blood-brain-barrier penetration based on LightGBM

<ins>Paper 3 Link</ins>: https://pubmed.ncbi.nlm.nih.gov/33112379/ 
 
## Key Terminology for Contextual Understanding: 

<ins>Lipinski Rule</ins>: a rule that states that for a drug to be effectively orally consumed, it needs to have the following properties 

    - Molecular weight: (<= 500 daltons) 

    - LogP: (<= 5) 

    - Hydrogen Bond Donors (HBD): (<= 5)

    - Hydrogen Bond Acceptors (HBA): (<= 10) 

<ins>Generic Algorithm (GA)</ins>: an optimization algorithm that finds molecules with high BBB permeability by using their molecular properties, such as their logBB value or their structure

<ins>Random Forest Model (RF)</ins>: an algorithm where decisions are split up into multiple trees based on different initial parameters or subsets of data, and combined to determine BBB permeability

<ins>Support Vector Machine (SVM)</ins>: an algorithm that quantifies the parameters into vectors and graphs it on a hyperplane (a plane that is one dimension less than the space dimension) to find the distance between the vector and the hyperplane, which measures the uncertainty value

<ins>Artificial Neural Network (ANN)</ins>: an algorithm that learns the weights (example: molecular features) essential to BBB permeability through adjustment of its value and outputs its permeability based on these features.

<ins>TP (true positive)</ins>: correct prediction of a BBB+ compound 

<ins>TN (true negative)</ins>: correct prediction of a BBB- compound

<ins>FP (false positive)</ins>: incorrect prediction of a BBB- compound as a BBB+

<ins>FT (false negative)</ins>: incorrect prediction of a BBB+ compound as a BBB- 

<ins>Gradient Boosting Decision Tree (GBDT)</ins>: a learning algorithm with an initial prediction where a sequence of multiple trees works to fix the previous errors of the prediction, which results in a more accurate prediction.

<ins>Exclusive Feature Bundling (EFB)</ins>: a technique where non-overlapping features are bundled into one value/feature, which decreases the overall number of features and time, increasing model efficiency.

<ins>Gradient-based one-sided sampling (GOSS)</ins>: a technique where small errors (small gradients) and big errors (large gradients) are chosen as a way to train for accuracy with the large gradients and time with the small gradients.
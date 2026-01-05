<p align="center">
  <img src="BioGA-X LOGO.png" width="80%" />
</p>


---------------------------------------
# BioGA-X: Explainable AI-Guided Genetic Algorithm for Biological Sequence Optimization
BioGA-X is a genetic algorithm framework tailored for **DNA and protein sequence engineering**. It leverages deep learning and explainable AI to efficiently navigate biological sequence spaces.  
XAI-Guided Operators: Uses **Integrated Gradients (IG)** to identify key residues, guiding mutations and crossovers to accelerate optimization.  
Fitness Evaluation: Integrates any **Neural Network** as a fitness function to predict biological activity.  
**Multi-Objective Optimization**: Employs **NSGA-II** to balance two competing goals:
  1. **Model Score**: Maximizing functional performance.
  2. **Seq Similarity**: Preserving evolutionary integrity (via Needleman-Wunsch).


## Workflow
 <img src="BioGA-X workflow.png" style="align:center" />

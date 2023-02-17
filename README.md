# Feating_Ensemble
Implementation of Feating (Feature Subspace Aggregating) ensemble.

Feating (Feature Subspace Aggregating) subdivides the feature-space into non-overlapping local regions in a single subdivision; and ensures that different subdivisions provide the distinct local neighborhoods for each point in the feature space. There are many ways a feature-space can be subdivided. Instead of using heuristics, Feating subdivides the feature-space exhaustively based on a user-specified number of features to control the level of localisation. The set of exhaustive feature-space subdivisions forms the basis to develop a new ensemble method which aggregates all local models or a random subset of these local models.

At first I choose 3 db satimage and segment and nursery then preprocessed the datasets

Second create combination with 1 and 2 and 3 from features for create trees in feating 

then calculate rank attribute that compute on information gain for all features and then rank from max to less

And after that with the result of combination we make all states

Then we create lvl tree for every single state of combinations

And at the end if our data were pure in leaves we place the label in that leaf and if not we place classifier in local Model leaves and get majority voting for leaves 

We use 3 classifier in that as known svm, j48, IBk , and for testing we go trough leaves and we predict our data with that local model were there



For more information regarding the Feating algorithm, please see the attached paper.


What does “Feating is an aggregation of local models” mean?
 Feating (Feature Subspace Aggregating) subdivides the feature-space into non-overlapping local regions in a single subdivision; and ensures that different subdivisions provide the distinct local neighborhoods for each point in the feature space. There are many ways a feature-space can be subdivided. Instead of using heuristics, Feating subdivides the feature-space exhaustively based on a user-specified number of features to control the level of localization

Why is localization important for us in classification?
Because subdivides the feature-space into non-overlapping local regions in a single subdivision
ensures that different subdivisions provide the distinct local neighborhoods for each point in the feature space
subdivides the feature-space exhaustively based on a user-specified number of features to control the level of localization

How does Level Tree perform feature space division? 
Subdivision for numeric attributes will be considered in the same manner as a cut-point selection in an ordinary decision tree using a heuristic such as information gain. As a result, though each attribute can appear at one level only in a Level Tree, the cut-points used for a numeric attribute on different nodes of the same level can be different.

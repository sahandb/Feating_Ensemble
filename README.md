# Feating_Ensemble
Implementation of Feating (Feature Subspace Aggregating) ensemble.

Feating (Feature Subspace Aggregating) subdivides the feature-space into non-overlapping local regions in a single subdivision; and ensures that different subdivisions provide the distinct local neighborhoods for each point in the feature space. There are many ways a feature-space can be subdivided. Instead of using heuristics, Feating subdivides the feature-space exhaustively based on a user-specified number of features to control the level of localisation. The set of exhaustive feature-space subdivisions forms the basis to develop a new ensemble method which aggregates all local models or a random subset of these local models.

For more information regarding the Feating algorithm, please see the attached paper.

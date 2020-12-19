# Applicability Domain

DA metric is defined as

$$ SDC = \sum_{i=1}^{n} \exp{-3TD_i / 1 - TD_i $$

where $$SDC$$ is the sum of the distance-weighted contributions gauges QSAR prediction accuracy, $$TD_i$$ is the Tanimoto distance (TD) between a target molecule and the ith training molecule, and n is the total number of training molecules. The $$TD$$ between two molecules is calculated by using the extended connectivity fingerprint with a diameter of four chemical bonds (ECFP_4). The TD value between two molecules ranges between 0 and 1the lower and upper limits corresponding to two molecules sharing all and no fingerprint features, respectively.


import lightgbm as lgb


def get_baked_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective='binary',
                              metric='binary_logloss',
                              class_weight='balanced',
                              random_state=0,
                              n_jobs=-1,
                              num_leaves=95,
                              n_estimators=320,
                              max_depth=14,
                              boosting_type='dart',
                              learning_rate=0.5,
                              min_split_gain=0,
                              min_child_weight=1e-05,
                              min_child_samples=30,
                              subsample=0.97175,
                              colsample_bytree=0.95,
                              subsample_freq=2,
                              reg_alpha=0,
                              reg_lambda=0)

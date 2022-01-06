from skopt.space import Real, Integer


RANDOM_STATE = 0
N_JOBS = -1

AUDIO_LIMIT = 5 * 22050  # 5 sec * sr
MEL_WINDOW_LENGTH = 8
N_MELS = 128
MFCC_WINDOW_LENGTH = 512
N_MFCC = 20
CHROMA_WINDOW_LENGTH = 32
N_CHROMA = 12


DAG_CONFIG = dict(
    # extract_features
    extract=dict(
        audio_limit=AUDIO_LIMIT,
        mel_window_length=MEL_WINDOW_LENGTH,
        n_mels=N_MELS,
        mfcc_window_length=MFCC_WINDOW_LENGTH,
        n_mfcc=N_MFCC,
        chroma_window_length=CHROMA_WINDOW_LENGTH,
        n_chroma=N_CHROMA
    ),

    # train_test_model
    test_size=0.2,
    random_state=RANDOM_STATE,

    # skopt/sklearn Pipeline
    standardize=dict(
        output_distribution='normal',
        random_state=RANDOM_STATE
    ),

    decomposition=dict(
        n_components=50,
        random_state=RANDOM_STATE
    ),

    oversample=dict(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    ),

    model=dict(
        objective='binary',
        metric='binary_logloss',
        class_weight='balanced',
        random_state=RANDOM_STATE,
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
        reg_lambda=0
    ),

    bayes_search=dict(
        n_iter=1,  # iterations of Bayesian optimization to run
        scoring='f1_weighted',
        n_jobs=-1,
        n_points=1,   # number of parameter settings to sample in parallel (per iteration)
        refit=True,  # after optimization, refits on entire dataset, so predictions can be made
        verbose=0,
        cv=3,
        search_spaces={
            # Power
            'model__num_leaves': Integer(90, 100),                                       # def=31
            'model__n_estimators': Integer(low=300, high=400),                           # def=100
            'model__max_depth': Integer(10, 28),                                         # def=-1 no limit

            # Learning
            'model__learning_rate': Real(low=0.1, high=0.9, prior='log-uniform'),             # def=0.1
            'model__min_child_weight': Real(low=0.000001, high=0.0001, prior='log-uniform'),  # def=1e-3
            'model__min_child_samples': Integer(30, 50),                                      # def=20

            # Bagging
            'model__subsample': Real(0.94, 0.98),                                             # def=1.0
            'model__colsample_bytree': Real(0.92, 0.999),                                     # def=1.0
        }
    )
)

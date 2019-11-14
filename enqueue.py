from typing import List

import luigi
import numpy as np

# - Para múltiplas GPUs:
## ./cuda_luigid --background --pidfile /tmp/luigid.pid --logdir /tmp/luigid_log
## PYTHONPATH="." ./cuda_luigi --module enqueue_pneumonia EnqueuePneumonia --seed 42 --workers 4
# - Para uma única GPU:
## PYTHONPATH="." luigi --module enqueue_pneumonia EnqueuePneumonia --seed 42 --local-scheduler
from recommendation.task.ifood import EvaluateIfoodModel, EvaluateIfoodCDAEModel, EvaluateIfoodAttCVAEModel, EvaluateIfoodTripletNetContentModel, EvaluateIfoodTripletNetWeightedModel
from recommendation.task.model.auto_encoder import UnconstrainedAutoEncoderTraining, AttentiveVariationalAutoEncoderTraining
from recommendation.task.model.triplet_net import TripletNetContentTraining, TripletNetTraining
from recommendation.task.model.matrix_factorization import MatrixFactorizationTraining
class EnqueueEvaluateCDAE(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter()

    def requires(self):
        for task_id in self.task_ids:
            yield EvaluateIfoodCDAEModel(
                model_module="recommendation.task.model.auto_encoder",
                model_cls="UnconstrainedAutoEncoderTraining",
                model_task_id=task_id,
            )

class EnqueueCDAE(luigi.WrapperTask):
    seed: int = luigi.IntParameter(default=42)
    batch_size_options: list = luigi.ListParameter(default=[50, 100, 200, 500, 1000])
    learning_rate_range: tuple = luigi.TupleParameter(default=(1e-5, 1e-3))

    def requires(self):
        random_state = np.random.RandomState(self.seed)

        for i in range(5):
            training_model = UnconstrainedAutoEncoderTraining(
                project="ifood_user_cdae",
                binary=True,
                optimizer="adam",
                learning_rate=random_state.uniform(self.learning_rate_range[0], self.learning_rate_range[1]),
                batch_size=int(random_state.choice(self.batch_size_options)),
                generator_workers=0,
                loss_function="focal",
                loss_function_params={"gamma": 10.0, "alpha": 1616.0},
                loss_wrapper="none",
                data_frames_preparation_extra_params={"split_per_user": True},
                gradient_norm_clipping=2.0,
                gradient_norm_clipping_type=1,
                data_transformation="support_based",
            )
            yield training_model


class EnqueueEvaluateAttentiveCVAE(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter(default=["AttentiveVariationalAutoEncoderTraining_selu__600_____146880a1ec","AttentiveVariationalAutoEncoderTraining_selu__600_____263553b2d1","AttentiveVariationalAutoEncoderTraining_selu__600_____491c066f34","AttentiveVariationalAutoEncoderTraining_selu__600_____59a3ab9647","AttentiveVariationalAutoEncoderTraining_selu__600_____6dd771b4a4","AttentiveVariationalAutoEncoderTraining_selu__600_____6fc4a5b213","AttentiveVariationalAutoEncoderTraining_selu__600_____85cf24edcd","AttentiveVariationalAutoEncoderTraining_selu__600_____99145a796b","AttentiveVariationalAutoEncoderTraining_selu__600_____a155d057ca","AttentiveVariationalAutoEncoderTraining_selu__600_____ab3701179d","AttentiveVariationalAutoEncoderTraining_selu__600_____cc1c8bd0ac","AttentiveVariationalAutoEncoderTraining_selu__600_____de95289241","AttentiveVariationalAutoEncoderTraining_selu__600_____e6824495a8","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____0723ddfc30","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____4fb23a7073","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____582c1e31b3","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____6e9398c04f","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____918a86d4b3","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____a9a8b72a48","AttentiveVariationalAutoEncoderTraining_sigmoid__600_____ed7ab4292e","AttentiveVariationalAutoEncoderTraining_tanh__600_____02dd8e3621","AttentiveVariationalAutoEncoderTraining_tanh__600_____2d77eb65fb","AttentiveVariationalAutoEncoderTraining_tanh__600_____3db754ed95","AttentiveVariationalAutoEncoderTraining_tanh__600_____3f98e7a344","AttentiveVariationalAutoEncoderTraining_tanh__600_____4524cd8410","AttentiveVariationalAutoEncoderTraining_tanh__600_____501b2fe657","AttentiveVariationalAutoEncoderTraining_tanh__600_____97d7c1a88d","AttentiveVariationalAutoEncoderTraining_tanh__600_____c706a492ba","AttentiveVariationalAutoEncoderTraining_tanh__600_____cf70ff3b3d","AttentiveVariationalAutoEncoderTraining_tanh__600_____d7900806f8","AttentiveVariationalAutoEncoderTraining_tanh__600_____eb0bc93eb9"])

    def requires(self):
        for task_id in self.task_ids:
            print(task_id)
            yield EvaluateIfoodAttCVAEModel(
                model_module="recommendation.task.model.auto_encoder",
                model_cls="AttentiveVariationalAutoEncoderTraining",
                model_task_id=task_id,
            )


class EnqueueAttentiveCVAE(luigi.WrapperTask):
    seed: int = luigi.IntParameter(default=42)
    batch_size_options: list = luigi.ListParameter(default=[50, 100, 200, 500, 1000])
    learning_rate_range: tuple = luigi.TupleParameter(default=(1e-5, 1e-3))
    activation_function_options: list = luigi.ListParameter(default=["selu", "relu", "tanh", "linear", "sigmoid"])
    vae_anneal_range: tuple = luigi.TupleParameter(default=(0.0, 1.0))
    attention_anneal_range: tuple = luigi.TupleParameter(default=(0.0, 1.0))
    dropout_prob_range: tuple = luigi.TupleParameter(default=(0.0, 0.9))
    optimizer_options: list = luigi.ListParameter(default=["adam", "rmsprop", "sgd", "adadelta", "adagrad", "adamax"])
    data_transformation_options: list = luigi.ListParameter(default=["support_based", "salt_and_pepper_noise", "none", "masking_noise"])

    def requires(self):
         
        random_state = np.random.RandomState(self.seed)

        for i in range(60):
            training_model = AttentiveVariationalAutoEncoderTraining(
                project="ifood_user_cdae",
                binary=False,
                optimizer=str(random_state.choice(self.optimizer_options)),
                learning_rate=random_state.uniform(self.learning_rate_range[0], self.learning_rate_range[1]),
                batch_size=int(random_state.choice(self.batch_size_options)),
                generator_workers=0,
                loss_function="attentive_vae_loss",
                loss_function_params={"anneal": random_state.uniform(self.vae_anneal_range[0], self.vae_anneal_range[1]),\
                 "anneal_att": random_state.uniform(self.attention_anneal_range[0], self.attention_anneal_range[1])},
                data_frames_preparation_extra_params={"split_per_user": True},
                data_transformation=str(random_state.choice(self.data_transformation_options)),
                activation_function=str(random_state.choice(self.activation_function_options)),
                dropout_module="dropout",
                dropout_prob=random_state.uniform(self.dropout_prob_range[0], self.dropout_prob_range[1]),
                encoder_layers=[600,200],
                decoder_layers=[200],
                attention_layers=[600],
            )
            yield training_model


class EnqueueTripletNetContent(luigi.WrapperTask):
    seed: int = luigi.IntParameter(default=42)
    batch_size_options: list = luigi.ListParameter(default=[128, 256, 512])
    word_embeddings_size_options: list = luigi.ListParameter(default=[128, 256, 512])
    reccurrence_hidden_size_options: list = luigi.ListParameter(default=[32, 64, 128, 256])
    word_embeddings_output_options: list = luigi.ListParameter(default=[128, 256])
    content_layers_options: list = luigi.ListParameter(default=[[512, 256], [512, 256, 128], [256, 128], [128], [512]])
    n_factors_options: list = luigi.ListParameter(default=[128, 256, 512])
    dropout_prob_range: tuple = luigi.TupleParameter(default=(0.0, 0.5))
    triplet_loss_options: list = luigi.ListParameter(default=['bpr_triplet', 'triplet_margin'])

    def requires(self):
        random_state = np.random.RandomState(self.seed)

        for i in range(45):
            training_model = TripletNetContentTraining(
                project="ifood_binary_buys_content_triplet_with_random_negative",
                batch_size=int(random_state.choice(self.batch_size_options)),
                loss_function=str(random_state.choice(self.triplet_loss_options)),
                word_embeddings_size=int(random_state.choice(self.word_embeddings_size_options)),
                recurrence_hidden_size=int(random_state.choice(self.reccurrence_hidden_size_options)),
                word_embeddings_output=int(random_state.choice(self.word_embeddings_output_options)),
                dropout_module="dropout",
                dropout_prob=random_state.uniform(self.dropout_prob_range[0], self.dropout_prob_range[1]),
                content_layers=list(random_state.choice(self.content_layers_options)),
                n_factors=int(random_state.choice(self.n_factors_options)),

            )

            yield training_model

class EnqueueEvaluateTripletNetContent(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter(default=["TripletNetContentTraining_selu____512_73cebd433f","TripletNetContentTraining_selu____256_8b05666af5","TripletNetContentTraining_selu____256_7b277793dd","TripletNetContentTraining_selu____256_6996cdaa1c","TripletNetContentTraining_selu____256_22c34c1db2","TripletNetContentTraining_selu____256_002f7e634d","TripletNetContentTraining_selu____128_6acf4b889a"])

    def requires(self):
        for task_id in self.task_ids:
            print(task_id)
            yield EvaluateIfoodTripletNetContentModel(
                model_module="recommendation.task.model.triplet_net",
                model_cls="TripletNetContentTraining",
                model_task_id=task_id,
            )

class EnqueueTripletNetTraining(luigi.WrapperTask):
    balance_options: list = luigi.ListParameter(default=[5, 25, 50, 150, 200, 250, 500, 1000, 5000, 10000])

    def requires(self):

        for i in self.balance_options:
            training_model = TripletNetTraining(
                project="ifood_buys_visits_triplet_with_random_negative",
                batch_size=512,
                loss_function="weighted_triplet",
                lr_scheduler_params={"step_size": 5, "gamma": 0.8},
                loss_function_params={"balance_factor": i},
                n_factors=2048,
                epochs=250,
                optimizer="adam",
                learning_rate=0.001,
                lr_scheduler="step",
            )

            yield training_model

# PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationTraining 
# --project ifood_binary_buys_and_buys_and_visits_cf_with_random_negative 
# --n-factors 256 --binary --loss-function implicit_feedback_bce --loss-function-params '{"confidence_weights": [1.0, 0.0004]}' 
# --metrics '["loss", "binary_accuracy", "precision", "recall", "f1_score"]' --negative-proportion 10 --epochs 100 
# --local-scheduler --batch-size 500 --save-item-embedding-tsv --save-user-embedding-tsv
class EnqueueMatrixFactorizationTraining(luigi.WrapperTask):
    n_factors: list = luigi.ListParameter(default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250])

    def requires(self):

        for n in self.n_factors:
            training_model = MatrixFactorizationTraining(
                project="ifood_binary_buys_and_buys_and_visits_cf_with_random_negative",
                batch_size=500,
                n_factors=n,
                binary=True,
                loss_function="implicit_feedback_bce",
                loss_function_params={"confidence_weights": [1.0, 0.0004]},
                lr_scheduler_params={"step_size": 5, "gamma": 0.8},
                metrics=["loss", "binary_accuracy", "precision", "recall", "f1_score"],
                negative_proportion=2,
                epochs=100,
            )

            yield training_model            

class EnqueueEvaluateTripletNetWeighted(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter(default=["TripletNetTraining____512____2d71bdc6d8","TripletNetTraining____512____37524fd14f","TripletNetTraining____512____3d67f3440a","TripletNetTraining____512____5c9517a5f1","TripletNetTraining____512____696eb4448e","TripletNetTraining____512____931fd9ee9f","TripletNetTraining____512____c3b2fe06b7","TripletNetTraining____512____eb375b71c2","TripletNetTraining____512____fe99d4b203"])

    def requires(self):
        for task_id in self.task_ids:
            print(task_id)
            yield EvaluateIfoodTripletNetWeightedModel(
                model_module="recommendation.task.model.triplet_net",
                model_cls="TripletNetContentTraining",
                model_task_id=task_id,
            )

class EnqueueEvaluateTripletNetWeighted(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter(default=["TripletNetTraining____512____2d71bdc6d8","TripletNetTraining____512____37524fd14f","TripletNetTraining____512____3d67f3440a","TripletNetTraining____512____5c9517a5f1","TripletNetTraining____512____696eb4448e","TripletNetTraining____512____931fd9ee9f","TripletNetTraining____512____c3b2fe06b7","TripletNetTraining____512____eb375b71c2","TripletNetTraining____512____fe99d4b203"])

    def requires(self):
        for task_id in self.task_ids:
            print(task_id)
            yield EvaluateIfoodTripletNetWeightedModel(
                model_module="recommendation.task.model.triplet_net",
                model_cls="TripletNetContentTraining",
                model_task_id=task_id,
            )

#PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler 
# --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorization 
# --model-task-id=MatrixFactorizationTraining____500_False_2cd769a409

class EnqueueEvaluateMatrixFactorization(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter(default=["MatrixFactorizationTraining____500_False_0e90d85c8a","MatrixFactorizationTraining____500_False_19373e938e","MatrixFactorizationTraining____500_False_1c1a702367","MatrixFactorizationTraining____500_False_2c7f4a044d","MatrixFactorizationTraining____500_False_30c3f60cac","MatrixFactorizationTraining____500_False_3350e5e066","MatrixFactorizationTraining____500_False_42fc8f88ae","MatrixFactorizationTraining____500_False_5cd44ca6a6","MatrixFactorizationTraining____500_False_67c409f796","MatrixFactorizationTraining____500_False_67e286bf36","MatrixFactorizationTraining____500_False_9afe2dc6f4","MatrixFactorizationTraining____500_False_a0b1e2f55e","MatrixFactorizationTraining____500_False_a1edeaaee6","MatrixFactorizationTraining____500_False_b23b60f223","MatrixFactorizationTraining____500_False_bbdfa30eba","MatrixFactorizationTraining____500_False_bda5d53fe4","MatrixFactorizationTraining____500_False_c1b0046e5a","MatrixFactorizationTraining____500_False_c8527dae21","MatrixFactorizationTraining____500_False_ca80f02d5a","MatrixFactorizationTraining____500_False_d26c2268dd","MatrixFactorizationTraining____500_False_d6335a426b","MatrixFactorizationTraining____500_False_df691e531c","MatrixFactorizationTraining____500_False_f1f920fa09","MatrixFactorizationTraining____500_False_f321a762d2","MatrixFactorizationTraining____500_False_f87ffed511","MatrixFactorizationTraining____500_False_ffe99b994d"])

    def requires(self):
        for task_id in self.task_ids:
            print(task_id)
            yield EvaluateIfoodModel(
                model_module="recommendation.task.model.matrix_factorization",
                model_cls="MatrixFactorizationTraining",
                model_task_id=task_id,
            )


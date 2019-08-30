from recommendation.task.model.matrix_factorization import MatrixFactorizationTraining, DeepMatrixFactorizationTraining
from recommendation.task.model.triplet_net import TripletNetTraining
from recommendation.task.model.auto_encoder import VariationalAutoEncoderTraining, UnconstrainedAutoEncoderTraining
from recommendation.task.data_preparation.base import WINDOW_FILTER_DF
from recommendation.task.ifood import EvaluateRandomIfoodModel, EvaluateMostPopularIfoodModel
import luigi
import multiprocessing


# PYTHONPATH="." luigi --module recommendation.task.model.all_models AllModels --local-scheduler
class AllModels(luigi.WrapperTask):
    #date = luigi.DateParameter(default=datetime.date.today())
    generator_workers: int = luigi.IntParameter(default=min(multiprocessing.cpu_count(), 20))
    batch_size: int = luigi.IntParameter(default=500)
    epochs: int = luigi.IntParameter(default=100)
    learning_rate: float = luigi.FloatParameter(1e-3)
    window_filter: str = luigi.ChoiceParameter(choices=WINDOW_FILTER_DF.keys(), default="one_week")

    def requires(self):

        
        for window_filter, _v in WINDOW_FILTER_DF.items():

          # Random
          yield EvaluateRandomIfoodModel(window_filter=window_filter, model_task_id one_week)

          # MostPopular
          yield EvaluateMostPopularIfoodModel(window_filter=window_filter, model_task_id one_week)

          # MatrixFactorizationTraining____500_False_c8c16debd0
          # PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationTraining --project ifood_binary_buys_cf_with_random_negative --n-factors 50 --binary --loss-function bce --metrics '["loss", "acc"]' --optimizer adam --learning-rate 1e-3 --batch-size 500 --pin-memory --local-scheduler
          yield MatrixFactorizationTraining(project='ifood_binary_buys_cf', 
                                            n_factors=50, binary=True, 
                                            loss_function='bce', 
                                            optimizer="adam",
                                            learning_rate=1e-3, 
                                            batch_size=500,
                                            epochs=self.epochs,
                                            generator_workers=self.generator_workers,
                                            window_filter=window_filter)

          # PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization DeepMatrixFactorizationTraining --project ifood_binary_buys_cf_with_random_negative --n-factors 50 --binary --loss-function bce --metrics '["loss", "acc"]' --optimizer adam --learning-rate 1e-3 --batch-size 500 --pin-memory --local-scheduler
          yield DeepMatrixFactorizationTraining(project='ifood_binary_buys_cf', 
                                            n_factors=50, binary=True, 
                                            loss_function='bce', 
                                            optimizer="adam",
                                            learning_rate=1e-3, 
                                            batch_size=500,
                                            epochs=self.epochs,
                                            generator_workers=self.generator_workers,
                                            window_filter=window_filter)

          # TripletNetTraining___500___a02717d114
          # PYTHONPATH="." luigi --module recommendation.task.model.triplet_net TripletNetTraining --project ifood_binary_buys_triplet_with_random_negative --n-factors 50 --metrics '["loss"]' --batch-size 500 --loss-function triplet_margin --pin-memory --local-scheduler
          yield TripletNetTraining(project='ifood_binary_buys_triplet_with_random_negative', 
                                  n_factors=50, 
                                  loss_function='triplet_margin', 
                                  optimizer="adam",
                                  learning_rate=1e-3, 
                                  batch_size=500,
                                  epochs=self.epochs,
                                  generator_workers=self.generator_workers,
                                  window_filter=window_filter)

          # PYTHONPATH="." luig
          # i --module recommendation.task.model.auto_encoder VariationalAutoEncoderTraining --project ifood_use
          # r_cdae --generator-workers=0 --local-scheduler --batch-size=500 --optimizer=adam --lr-scheduler=step
          #  --lr-scheduler-params='{"step_size": 5, "gamma": 0.8}' --dropout-module=dropout --dropout-prob=0.5
          # --activation-function=tanh --epochs=200 --window-filter=all --encoder-layers=[600,200] --decoder-lay
          # ers=[600] --loss-function-params='{"anneal": 0.0}'        
          yield VariationalAutoEncoderTraining(project='ifood_user_cdae', 
                                  optimizer="adam",
                                  learning_rate=1e-3, 
                                  batch_size=500,
                                  lr_scheduler='step',
                                  lr_scheduler_params={"step_size": 5, "gamma": 0.8},
                                  dropout_prob=0.5,
                                  dropout_module='dropout',
                                  activation_function='tanh',
                                  encoder_layers=[600,200],
                                  decoder_layers=[600],
                                  loss_function_params={"anneal": 0.0},
                                  epochs=self.epochs,
                                  generator_workers=self.generator_workers,
                                  window_filter=window_filter)        

          # PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder UnconstrainedAutoEncoderTraining --project ifood_user_cdae --local-scheduler  --generator-workers=0 --optimizer=adagrad --dropout-module=dropout --dropout-prob=0.8 --learning-rate=0.001 --epochs=20 --test-size=0.02 --val-size=0.02 --activation-function=selu --batch-size=2048
          yield UnconstrainedAutoEncoderTraining(project='ifood_user_cdae', 
                                  optimizer="adagrad",
                                  learning_rate=1e-3, 
                                  batch_size=2048,
                                  dropout_prob=0.8,
                                  dropout_module='dropout',
                                  activation_function='selu',
                                  epochs=self.epochs,
                                  generator_workers=self.generator_workers,
                                  window_filter=window_filter)    
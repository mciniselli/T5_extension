# An empirical study on the usage of T5 models for Code Completion

In this work we explored the capabilities of **Text-To-Text Transfer Transformer (T5)** for Code Completion related tasks. For the task we're going to mask some tokens and then check if the model is able to predict them.

### Preliminary step
The training of the model is done on a TPU instance of **Colab**.
A GCS Bucket is mandatory.
To Set up a new GCS Bucket for training and fine-tuning a T5 Model, please follow the original guide provided by Google [here](https://cloud.google.com/storage/docs/quickstart-console).


### Pipeline
* ##### Dataset

    You can find the datasets used for pretraining and fine-tuning the models [here](https://drive.google.com/drive/folders/17LlqNQeZ6BkRACJY34munsHtymOqLunV?usp=sharing) and [here](https://drive.google.com/drive/folders/1_tSaJzoG3v9ypwmI3HzR2GTbWgWgZJ-L?usp=sharing). 
    For pretraining tokenizer we shared also **key.csv** file with information about each method.
* ##### Tokenizer

    We trained the tokenizer using the script in `Tokenizer` folder.
    ```
    python3 tokenizer.py --input=<input_file> --model_prefix=code --vocab_size=32000 --bos_id=-1  --eos_id=1 --unk_id=2 --pad_id=0
    ```
    Where:
    - input: the path for the txt file containing the code to tokenize 
    - model_prefix: the prefix for the tokenizer (e.g. code => it generates code.vocab and code.model)
    - vocab_size: the size of the vocabulary
    - bos_id: begin of sentence id (this change only the order or the tokens stored in the vocabulary- bos_id: begin of sentence id (this change only the order or the tokens stored in the vocabulary)
    - eos_id: end of sentence id (this change only the order or the tokens stored in the vocabulary)
    - unk_id: unknown token id (this change only the order or the tokens stored in the vocabulary)
    - pad_id: padding id (this change only the order or the tokens stored in the vocabulary)
    You can find the tokenizer in `Pretraining/tokenizer_model` folder.
    
* ##### Pretraining
    
    For the pretraining model you can find the notebook **pretrain.ipynb** in `Pretraining` folder. 
    The notebook has some comments that explain how to run it.
    You can also find the gin file for config in `configuration_file` folder and the trained tokenizer in `tokenizer_model` folder.
    The pretrained model is available [here](https://drive.google.com/drive/folders/1783WqX5GypthAG9FS2uYlSDR9d0zslEB?usp=sharing)
* ##### Hyper Parameter tuning

    We did hyper parameter tuning to find the best model for the finetuning.
    We tested 4 configuration and trained the model for 100k steps.
    The configuration are the following:
    - constant learning rate (lr = 0.001)
    - Inverse Square Root (warmup_steps = 10000)
    - slanted (cut_fraction=0.1, ratio=32, max_learning_rate=0.01, start_step=0)
    - polynomial learning rate (starter_learning_rate=0.01, end_learning_rate=1e-6, decay_step=10000, power=0.5)
    
    You can find the commented notebooks in `HP_Tuning/pretraining_script`.
    The configuration files for each HP tuning are in `HP_Tuning/configuration_files`.
    You can find the script to evaluate the performances in `HP_Tuning/evaluation` folder.
    ```
    python3 perfect_predictions.py --folder <folder_with_prediction> 
    ```
    In the **--folder** you have to save all the files generated during the evaluation by tensorflow.
    You can find the files [here](https://drive.google.com/drive/folders/1HoqMM1adk7AiLknvc42ErjGcpgQn9jiM?usp=sharing) the HP tuning models and the files for the predictions
    
    Then we evaluated the performance; the best model was **slanted**.
    Here the **percentage of perfect predictions** for each model:
    | DATASET           | CONSTANT | SLANTED | ISR   | POLYNOMIAL |
    |-------------------|----------|---------|-------|------------|
    | java construct    |    50.51 |   52.11 | 50.77 |      31.36 |
    | java block        |    24.85 |   26.92 | 25.52 |       7.46 |
    | java token        |    65.42 |   66.45 | 65.43 |      44.75 |
    | android construct |    48.20 |   49.98 | 48.59 |      27.98 |
    | android block     |    25.97 |   27.96 | 26.46 |       7.99 |
    | android token     |    68.23 |   69.37 | 68.38 |      46.70 |
    | overall           |    57.62 |   58.97 | 57.81 |      37.28 |
    
* ##### Finetuning

    For the finetuning phase, we wanted to evaluate if the pretrained model is able to increase the performance of the model and if the training on multiple tasks can give reciprocal benefits to all the tasks.
    To **evaluate the performance** of each model we used a beam size of 1 (in order to be comparable with RoBERTa model).
    We did 3 different fine tuning:
    - A multi-task finetuning (in `Finetuning/multitask` folder)
    - A single-task finetuning for each dataset (6 models) starting from pretrained model (in `Finetuning/single_task_from_pretrained` folder)
    - A single-task finetuning for each dataset (6 models) starting from scratch (in `Finetuning/single_task_no_pretrained` folder)
    
    We finetuned the **multi-task** model for 400k steps (around 29 epochs).
    We chosed the number of steps of the other models so that the number of training epochs is 29.
    The following table contains the number of training steps for each model:
    | DATASET           | STEPS |
    |-------------------|------:|
    | java construct    | 85000 |
    | java block        | 34000 |
    | java token        | 85000 |
    | android construct | 85000 |
    | android block     | 24000 |
    | android token     | 85000 |
    
    You can finetune and evaluation running **Fine_tuning.ipynb** and **evaluate.ipynb** notebooks (read the comments in the notebook).
    For the evaluation you have to load on the Bucket the input file containing the methods you want to predict and use the path of this file for the **input_file** in the predict method.
    For the multi-task finetuning you have to merge all the input files for each task in order to predict all methods in one single step
    
    **Multi-task finetuning**
    
    You can evaluate the **number of perfect predictions** running:
    ```
    python3 perfect_predictions.py --input_path <path_to_input_file>  --target_path <path_to_target_file> --prediction_path <path_to_prediction_file>
    ```
    Where
    - input_path contains the file you want to predict
    - target_path contains the file with the correct value that the model should predict
    - prediction_path contains the file with the T5 predictions
    
    The performance of the model on each task is the following:
    | DATASET           | PERFECT PREDICTIONS | TOTAL NUMBER OF RECORDS | PERCENTAGE PERFECT PREDICTION |
    |-------------------|---------------------|-------------------------|-------------------------------|
    | java construct    |               56297 |                  106237 |                         52.99 |
    | java block        |               11537 |                   40008 |                         28.84 |
    | java token        |              145540 |                  219486 |                         66.31 |
    | android construct |               51053 |                  100536 |                         50.78 |
    | android block     |                8024 |                   26978 |                         29.74 |
    | android token     |              138877 |                  200504 |                         69.26 |
    | overall           |              411328 |                  693749 |                         59.29 |
    
    **Single-task finetuning from pretrained model**
    
    You can evaluate the **number of perfect predictions** running:
    ```
    python3 perfect_predictions.py --target_path <path_to_target_file> --prediction_path <path_to_prediction_file>
    ```
    Where
    - target_path contains the file with the correct value that the model should predict
    - prediction_path contains the file with the T5 predictions

    The performance of the models is the following:
    | DATASET           | PERFECT PREDICTIONS | TOTAL NUMBER OF RECORDS | PERCENTAGE PERFECT PREDICTION |
    |-------------------|---------------------|-------------------------|-------------------------------|
    | java construct    |               54394 |                  106237 |                         51.20 |
    | java block        |               10873 |                   40008 |                         27.18 |
    | java token        |              137967 |                  219486 |                         62.86 |
    | android construct |               49567 |                  100536 |                         49.30 |
    | android block     |                7413 |                   26978 |                         27.48 |
    | android token     |              129990 |                  200504 |                         64.83 |
    | overall           |              390204 |                  693749 |                         56.24 |
    
    **Single-task finetuning from scratch**
    
    You can evaluate the **number of perfect predictions** running:
    ```
    python3 perfect_predictions.py --target_path <path_to_target_file> --prediction_path <path_to_prediction_file>
    ```
    Where
    - target_path contains the file with the correct value that the model should predict
    - prediction_path contains the file with the T5 predictions
    
    The performance of the models is the following:
    
    | DATASET           | PERFECT PREDICTIONS | TOTAL NUMBER OF RECORDS | PERCENTAGE PERFECT PREDICTION |
    |-------------------|---------------------|-------------------------|-------------------------------|
    | java construct    |               51429 |                  106237 |                         48.41 |
    | java block        |                9162 |                   40008 |                         22.90 |
    | java token        |              133716 |                  219486 |                         60.92 |
    | android construct |               47001 |                  100536 |                         46.75 |
    | android block     |                6140 |                   26978 |                         22.76 |
    | android token     |              127871 |                  200504 |                         63.77 |
    | overall           |              375319 |                  693749 |                         54.10 |
    
    A recap table with the percentage of perfect prediction is below:
    | DATASET           | MULTI-TASK PRETRAINED | SINGLE-TASK PRETRAINED | SINGLE-TASK FROM SCRATCH |
    |-------------------|-----------------------|------------------------|--------------------------|
    | java construct    |                 52.29 |                  51.20 |                    48.41 |
    | java block        |                 28.84 |                  27.18 |                    22.90 |
    | java token        |                 66.31 |                  62.86 |                    60.92 |
    | android construct |                 50.78 |                  49.30 |                    46.75 |
    | android block     |                 29.74 |                  27.48 |                    22.76 |
    | android token     |                 69.26 |                  64.83 |                    63.77 |
    | overall           |                 59.29 |                  56.24 |                    54.10 |
    
    You can see that training a model with multiple tasks is beneficial for all the tasks.
    The pretraining is useful to increase the performances of each model.
    
    You can find the models and the prediction [here](https://drive.google.com/drive/folders/1tRsKzKvcmJRaczOUzYHmhIlR8WZR38qY?usp=sharing)
    
### Score Analysis
We chose the **best model** found (i.e., multi-task model with pre-training) to further analyze its performance. 
T5 returns a score for each prediction, ranging from minus infinity to 0. This score is the log likelihood of the prediction itself. It means that if the score is 0 then the log likelihood (ln) of the prediction is 0. So the likelihood is 1 and this implies that the model has confidence 100\% that the prediction is correct (high confidence). If the score is -2 then the log likelihood of the prediction is -2. So the likelihood is 0.14 and this implies that the model has confidence 14\% that the prediction is correct (low confidence).

We wanted to check if the model has **high confidence** when the prediction is correct. If this is true, then we could suggest a prediction to the developer only when the confidence is higher than a specific threshold.
The likelihood goes from 0 to 1, so we split the confidence of the model in ten classes with a breadth of 0.1 (class 1 has a confidence c so that 0<=c<0.1, class 2 has a confidence of 0.1<=c<0.2 and so on).

The following table reports the percentage of perfect predictions for each class:
| CLASS | PERFECT PREDICTIONS | TOTAL NUMBER OF PREDICTIONS | PERFECT PREDICTIONS (\%) |
|-------|---------------------|-----------------------------|-------------------------|
| 1     |               1,011 |                      82,440 |                    1.23 |
| 2     |               1,234 |                      28,009 |                    4.41 |
| 3     |               2,365 |                      26,637 |                    8.88 |
| 4     |               4,108 |                      26,305 |                   15.62 |
| 5     |               7,054 |                      28,627 |                   24.64 |
| 6     |              11,891 |                      33,223 |                   35.79 |
| 7     |              14,034 |                      32,258 |                   43.51 |
| 8     |              18,243 |                      35,043 |                   52.06 |
| 9     |              30,279 |                      47,826 |                   63.31 |
| 10    |             321,109 |                     353,381 |                   90.87 |   

As you can see from the table, if the confidence is greater than 0.9 we have **more than 90\% of perfect predictions**. Moreover, 321,109 perfect predictions out of 411,328 total perfect predictions (78\%) have a confidence greater than 0.9. This suggests that a developer can confidently use our model when the prediction’s confidence is greater than 0.9. We also checked the same metrics for each test dataset in isolation and the percentage of perfect predictions of each class was similar.

Then, we checked if the predictions that are **not correct** for each class **are similar** to the target prediction. The idea is to check if when the model has a high confidence and the prediction is not correct, the difference is small (maybe the only different token is “<” instead of “<=”). If this is True, we know that all the predictions, although not perfect, are good when the confidence is high.
We measured the mean and the median Levenshtein distance (divided by the maximum number of tokens among the prediction and the target so that the value is rescaled in [0,1] interval) among all classes (we used the number of tokens retrieved from the abstract dataset, since src2abs is able to split each method in abstract tokens); the results are the following:
| CLASS | MEAN | MEDIAN |
|:-----:|:----:|:------:|
|   1   | 0.50 |  0.50  |
|   2   | 0.45 |  0.40  |
|   3   | 0.45 |  0.40  |
|   4   | 0.45 |  0.40  |
|   5   | 0.45 |  0.40  |
|   6   | 0.45 |  0.40  |
|   7   | 0.46 |  0.423 |
|   8   | 0.46 |  0.40  |
|   9   | 0.47 |  0.44  |
|   10  | 0.50 |  0.50  |

There is **no difference** in mean or median among the classes, so our hypothesis is not verified. The number of tokens that we need to remove, add or change from each class is the same.

Finally, we verified to what extent for the model **it is easier to predict the shorter predictions** (the confidence then is higher). There is a difference in the average number of tokens when the method is correctly predicted, compared with the cases when the prediction is not correct.
In the table below we reported for each class the average number of tokens for all methods and the average number of tokens for the methods correctly and wrongly predicted. Just as an example, looking at the first row, we can see that the prediction of the methods with a confidence lower than 0.1 has on average 6.95 tokens. If we consider only the perfect prediction, the average is 6.23 (so the shortest methods are, in general, predicted more correctly). If we focus on the wrongly predicted methods, their average length is 6.96 tokens.
| CLASS | AVERAGE TOKENS | AVERAGE TOKENS FOR PERFECT PREDICTION | AVERAGE TOKENS FOR NON PERFECT PREDICTIONS |
|-------|----------------|---------------------------------------|--------------------------------------------|
| 1     |           6.95 |                                  6.23 |                                       6.96 |
| 2     |           7.06 |                                  6.55 |                                       7.08 |
| 3     |           6.75 |                                  6.10 |                                       6.81 |
| 4     |           6.42 |                                  5.56 |                                       6.58 |
| 5     |           6.05 |                                  5.16 |                                       6.33 |
| 6     |           5.60 |                                  4.58 |                                       6.16 |
| 7     |           5.33 |                                  4.44 |                                       6.02 |
| 8     |           5.08 |                                  4.30 |                                       5.91 |
| 9     |           4.71 |                                  4.07 |                                       5.81 |
| 10    |           3.15 |                                  2.91 |                                       5.52 |

As we can see, the prediction with the highest confidence (greater than 0.9) has on average 3.15 tokens, whilst for the methods with the lowest confidence (less than 0.1) the average length is 6.95 tokens.

We also evaluated **BLEU score** and **Levenshtein distance** in order to compare T5 model with RoBERTa.
You can find the results in the paper.

You can find all the scripts in `Score` folder.
**evaluate_score.ipynb** allows you to create the score for each prediction (you have to pass as input and target the input file and the prediction generated by T5 model).

To analyze the percentage of perfect prediction and the difference in Levenshtein distance among all classes you can run:
```
python3 src/6_evaluate_score/score_analysis.py --input_path inp/6_evaluate_score --score_path out/6_evaluate_score --score
```


   
    
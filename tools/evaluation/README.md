# FWA_evaluator

- This script calculates scores for FWA tasks.
- Tested with Python 3.11.9.
- You may need to pip install nltk.

## Usage

### Calculate scores

```bash
python run-evaluator.py (result_path) (GT_path) output

Arguments:
    arg[1]: Path to the evaluation target execution log JSON folder or file (recursively searches for .log files in the specified path)
    arg[2]: Path to the ground-truth JSON folder or file (recursively searches for .json files in the specified path)
    arg[3]: Output JSON file path for evaluation results; outputs the following two files
      A. JSON file - Accuracy rate for each task category
      B. TXT file - Detailed judgment results for each task, with "_detail.txt" appended to A
```
      
### Classify categories

```bash
python calc_category.py (category_file) (score_txt) > output.txt

Arguments:
  arg[1]: CSV file containing category definitions (./Sample_Data/category_[factory,retail,warehouse].csv)
  arg[2]: Detailed judgment results file (_detail.txt)

Output:
  "Category", "Number of Tasks", "Total Scores", and "Average Score" are shown in tab-delimited format.

Category ID	Task Count	Total Score	Average Score
------------------------------------------------------------
01_abstract_documents	9	5.0000	0.5556
01_abstract_images	10	7.0000	0.7000
01_abstract_videos	10	3.0000	0.3000
01_spatiotemporal_images	44	2.0000	0.0455
01_spatiotemporal_videos	34	10.0000	0.2941
01_temporal_videos	17	2.0000	0.1176
02_make_decision	16	4.0000	0.2500
03_reporting	28	4.0000	0.1429
```



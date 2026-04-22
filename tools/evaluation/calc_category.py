import csv
import argparse


CATEGORY_ABSTRACT_DOCUMENTS={
    "name": "01_abstract_documents",
    "ids": [
        "2.2.1"
    ]
}

CATEGORY_ABSTRACT_IMAGES={
    "name": "01_abstract_images",
    "ids": [
        "2.3.1.2.1",
        "2.3.2.1.1",
        "2.3.2.2.1"
    ]
}

CATEGORY_ABSTRACT_VIDEOS={
    "name": "01_abstract_videos",
    "ids": [
        "2.3.1.2.2",
        "2.3.2.1.2",
        "2.3.2.2.2",
        "2.3.2.3.2"
    ]
}

CATEGORY_TEMPORAL_VIDEOS={
    "name": "01_temporal_videos",
    "ids": [
        "2.2.3"
    ]
}

CATEGORY_SPATIOTEMPORAL_IMAGES={
    "name": "01_spatiotemporal_images",
    "ids": [
        "2.3.3.1.1",
        "2.3.3.2.1"
    ]
}

CATEGORY_SPATIOTEMPORAL_VIDEOS={
    "name": "01_spatiotemporal_videos",
    "ids": [
        "2.3.3.1.2",
        "2.3.3.2.2"
    ]
}

CATEGORY_MAKEDICISION={
    "name": "02_make_decision",
    "ids": [
        "2.4.1",
        "2.4.1.3",
        "2.4.2",
        "2.4.2.3",
    ]
}

CATEGORY_REPORTING={
    "name": "03_reporting",
    "ids": [
        "2.5.1"
    ]
}


SUB_CATEGORIES = [
    CATEGORY_ABSTRACT_DOCUMENTS,
    CATEGORY_ABSTRACT_IMAGES,
    CATEGORY_ABSTRACT_VIDEOS,

    CATEGORY_TEMPORAL_VIDEOS,

    CATEGORY_SPATIOTEMPORAL_IMAGES,
    CATEGORY_SPATIOTEMPORAL_VIDEOS,

    CATEGORY_MAKEDICISION,
    CATEGORY_REPORTING
]



def load_category_mapping(csv_file):
    """Load task ID to category ID mapping from CSV file"""
    category_map = {}
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                task_id = row[0].strip()
                category_id = row[1].strip()
                category_map[task_id] = category_id
    return category_map

def load_scores(txt_file):
    """Load task ID and scores from text file"""
    scores = {}
    with open(txt_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split('\t')  # Assuming tab-separated
            if len(parts) >= 6:
                task_id = parts[0].strip()
                try:
                    score = float(parts[5].strip())
                    scores[task_id] = (score > 0.5)
                except ValueError:
                    print('error parsing score for task:', task_id)
                    continue
    return scores

def calculate_category_stats(category_map, scores):
    """Calculate statistics for each category"""
    category_stats = {}

    for sub_cat in SUB_CATEGORIES:
        sub_cat_id = sub_cat['name']
        category_stats[sub_cat_id] = {
            'task_count': 0,
            'total_score': 0.0,
            'scores': [],
            'average_score': 0.0
        }
    
    for task_id, score in scores.items():
        if task_id in category_map:
            category_id = category_map[task_id]
            
            if category_id not in category_stats:
                category_stats[category_id] = {
                    'task_count': 0,
                    'total_score': 0.0,
                    'scores': [],
                    'average_score': 0.0
                }
            
            category_stats[category_id]['task_count'] += 1
            category_stats[category_id]['total_score'] += score
            category_stats[category_id]['scores'].append(score)

            # sub tasks
            for sub_cat in SUB_CATEGORIES:
                if category_id in sub_cat['ids']:
                    sub_cat_id = sub_cat['name']
                    category_stats[sub_cat_id]['task_count'] += 1
                    category_stats[sub_cat_id]['total_score'] += score
                    category_stats[sub_cat_id]['scores'].append(score)
                
        else:
            print('task id not found in category map:', task_id,(category_map.get(task_id)))
    
    # Calculate average scores
    for category_id in category_stats:
        stats = category_stats[category_id]
        if stats['task_count'] > 0: 
            stats['average_score'] = stats['total_score'] / stats['task_count']
    
    return category_stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate category statistics from task scores')
    parser.add_argument('csv_file', help='CSV file containing task ID to category ID mapping')
    parser.add_argument('txt_file', help='Text file containing task scores')
    args = parser.parse_args()
    
    # File paths from arguments
    csv_file = args.csv_file
    txt_file = args.txt_file
    
    # Load data
    category_map = load_category_mapping(csv_file)
    scores = load_scores(txt_file)

    #print(f"Loaded {len(category_map)} task-category mappings.")
    #print(scores)
    
    # Calculate statistics
    category_stats = calculate_category_stats(category_map, scores)

    # Output results
    print("Category ID\tTask Count\tTotal Score\tAverage Score")
    print("-" * 60)
    
    for category_id in sorted(category_stats.keys()):
        # Skip categories with comma-separated numeric IDs
        if '.' in category_id:
            continue
        stats = category_stats[category_id]
        print(f"{category_id}\t{stats['task_count']}\t{stats['total_score']:.4f}\t{stats['average_score']:.4f}")

if __name__ == "__main__":
    main()
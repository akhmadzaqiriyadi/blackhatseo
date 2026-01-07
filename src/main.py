import sys
import argparse
import logging
import json
import os
import pandas as pd
from .config import load_config, setup_logging, MODELS_DIR, DATA_DIR
from .scraper import WebScraper
from .builder import DatasetBuilder
from .detector import BlackHatSEODetector

logger = setup_logging("app.log", verbose=False) # Will be reconfigured in main

def test_in_jupyter():
    """Test detector in Jupyter."""
    logger.info("Running Jupyter test...")
    scraper = WebScraper(use_selenium=False)
    logger.info("Testing dataset building...")
    builder = DatasetBuilder(scraper)
    clean_queries = ["berita indonesia", "pendidikan indonesia"]
    blackhat_queries = ["judi online", "togel online"]
    builder.build_dataset(
        clean_queries=clean_queries,
        blackhat_queries=blackhat_queries,
        output_path=DATA_DIR / "test_train_data.csv",
        threshold=5
    )
    detector = BlackHatSEODetector(model_path=MODELS_DIR / 'model.pkl', vectorizer_path=MODELS_DIR / 'vectorizer.pkl', use_selenium=False)
    urls = ['https://example.com']
    results = detector.predict(urls)
    for result in results:
        if 'error' in result:
            logger.error(f"{result['url']}: {result['error']}")
        else:
            prediction = "Black Hat SEO" if result['prediction'] == 1 else "Clean"
            logger.info(f"{result['url']}: {prediction} (confidence: {result['probability']*100:.2f}%)")
    explanation = detector.explain_prediction(urls[0])
    logger.info(f"\nExplanation for {explanation['url']}:")
    logger.info(f"Prediction: {explanation['prediction']}")
    logger.info(f"Confidence: {explanation['confidence']}")
    logger.info("Reasons:")
    for reason in explanation['reasons']:
        logger.info(f"- {reason}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Black Hat SEO Detection Tool')
    parser.add_argument('--config', help='Path to config JSON file', default=None)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--train-urls', required=True, help='CSV file with URLs and labels')
    train_parser.add_argument('--model-output', default=str(MODELS_DIR / 'model.pkl'), help='Path to save model')
    train_parser.add_argument('--vectorizer-output', default=str(MODELS_DIR / 'vectorizer.pkl'), help='Path to save vectorizer')
    
    predict_parser = subparsers.add_parser('predict', help='Predict black hat SEO')
    predict_parser.add_argument('--urls', required=True, help='Text file with URLs')
    predict_parser.add_argument('--model', default=str(MODELS_DIR / 'model.pkl'), help='Path to trained model')
    predict_parser.add_argument('--vectorizer', default=str(MODELS_DIR / 'vectorizer.pkl'), help='Path to trained vectorizer')
    predict_parser.add_argument('--output', help='Path to save results as JSON')
    
    explain_parser = subparsers.add_parser('explain', help='Explain prediction')
    explain_parser.add_argument('--url', required=True, help='URL to explain')
    explain_parser.add_argument('--model', default=str(MODELS_DIR / 'model.pkl'), help='Path to trained model')
    explain_parser.add_argument('--vectorizer', default=str(MODELS_DIR / 'vectorizer.pkl'), help='Path to trained vectorizer')
    
    build_parser = subparsers.add_parser('build-dataset', help='Build training dataset')
    build_parser.add_argument('--clean-queries', help='File with clean search queries (one per line)')
    build_parser.add_argument('--blackhat-queries', help='File with black hat search queries (one per line)')
    build_parser.add_argument('--url-files', nargs='*', help='Text files with URLs')
    build_parser.add_argument('--output', default=str(DATA_DIR / 'train_data.csv'), help='Output CSV file')
    build_parser.add_argument('--threshold', type=int, default=5, help='Keyword count threshold for labeling')
    build_parser.add_argument('--serpapi-key', help='SerpApi key for search', default='7f0a9dd1fa51513c901923823c413d94906c75b0f171cdce302706da08c97ec2')
    
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging("app.log", verbose=args.verbose or config.get('verbose', False))
    
    logger.debug(f"Loaded config: {config}")
    
    scraper = WebScraper(use_selenium=config.get('use_selenium', False))
    
    if args.command == 'train':
        train_path = DATA_DIR / args.train_urls if not pd.io.common.is_url(args.train_urls) and not os.path.exists(args.train_urls) else args.train_urls
        # If train_path is just a filename, assume it's in data dir
        if not os.path.exists(train_path) and os.path.exists(DATA_DIR / args.train_urls):
            train_path = DATA_DIR / args.train_urls

        try:
            train_df = pd.read_csv(train_path)
        except Exception as e:
            logger.error(f"Could not read CSV file {train_path}: {e}")
            sys.exit(1)

        if 'url' not in train_df.columns or 'label' not in train_df.columns:
            logger.error("Input CSV must contain 'url' and 'label' columns.")
            sys.exit(1)
            
        if len(train_df) < 10:
            logger.warning("Dataset is very small (<10 URLs). Training may be unreliable.")
            
        detector = BlackHatSEODetector(use_selenium=config.get('use_selenium', False), use_bert=config.get('use_bert', False))
        detector.train_model(train_df['url'].tolist(), train_df['label'].tolist())
        detector.save_model(args.model_output, args.vectorizer_output)
        
    elif args.command == 'predict':
        urls_path = DATA_DIR / args.urls if not os.path.exists(args.urls) and os.path.exists(DATA_DIR / args.urls) else args.urls
        try:
            with open(urls_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Could not read URLs file {urls_path}: {e}")
            sys.exit(1)

        detector = BlackHatSEODetector(args.model, args.vectorizer, config.get('use_selenium', False), config.get('use_bert', False))
        results = detector.predict(urls)
        
        if args.output:
            output_path = DATA_DIR / args.output if not os.path.isabs(args.output) else args.output
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            for result in results:
                if 'error' in result:
                    logger.error(f"{result['url']}: {result['error']}")
                else:
                    prediction = "Black Hat SEO" if result['prediction'] == 1 else "Clean"
                    logger.info(f"{result['url']}: {prediction} (confidence: {result['probability']*100:.2f}%)")
                    
    elif args.command == 'explain':
        detector = BlackHatSEODetector(args.model, args.vectorizer, config.get('use_selenium', False), config.get('use_bert', False))
        explanation = detector.explain_prediction(args.url)
        logger.info(f"URL: {explanation['url']}")
        logger.info(f"Prediction: {explanation['prediction']}")
        logger.info(f"Confidence: {explanation['confidence']}")
        logger.info("Reasons:")
        for reason in explanation['reasons']:
            logger.info(f"- {reason}")
        if 'top_keywords' in explanation:
            logger.info("Top Keywords:")
            for keyword, count in explanation['top_keywords']:
                logger.info(f"- {keyword}: {count}")
                
    elif args.command == 'build-dataset':
        builder = DatasetBuilder(scraper, serpapi_key=args.serpapi_key)
        clean_queries = []
        blackhat_queries = []
        
        if args.clean_queries:
            cq_path = DATA_DIR / args.clean_queries if not os.path.exists(args.clean_queries) and os.path.exists(DATA_DIR / args.clean_queries) else args.clean_queries
            with open(cq_path, 'r') as f:
                clean_queries = [line.strip() for line in f if line.strip()]
                
        if args.blackhat_queries:
            bhq_path = DATA_DIR / args.blackhat_queries if not os.path.exists(args.blackhat_queries) and os.path.exists(DATA_DIR / args.blackhat_queries) else args.blackhat_queries
            with open(bhq_path, 'r') as f:
                blackhat_queries = [line.strip() for line in f if line.strip()]
                
        # Handle url_files similarly
        url_files = []
        if args.url_files:
            for uf in args.url_files:
                 uf_path = DATA_DIR / uf if not os.path.exists(uf) and os.path.exists(DATA_DIR / uf) else uf
                 url_files.append(str(uf_path))

        builder.build_dataset(
            clean_queries=clean_queries,
            blackhat_queries=blackhat_queries,
            url_files=url_files,
            output_path=args.output,
            threshold=args.threshold
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        test_in_jupyter()
    else:
        main()

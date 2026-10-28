import torch
from torch_geometric.loader import NeighborLoader
from sample_utils import get_train_data, calc_metrics, get_test_data
from utils import *
import pandas as pd
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label every user in <data-root>/<dataset>/processed_data as bot (1) or human (0).')
    parser.add_argument('--model', required=True, help='path to a trained checkpoint (.pth)')
    parser.add_argument('--dataset', default='US-24', help='folder name under --data-root')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='artifacts/predictions/test_predictions.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    logger = set_logger('test.log', exp_name="load&eval", root='artifacts/logs/')
    logger.info("Starting test.py")

    device = args.device

    # Load test model
    model = torch.load(args.model, map_location=device)
    model.eval().to(device)

    logger.info("Model loaded and set to evaluation mode")

    # Load data and build test loader
    data = get_test_data(args.dataset, root=args.data_root)
    test_loader = NeighborLoader(
        data,
        num_neighbors=[256] * 2,
        batch_size=1024,
        input_nodes=data.test_idx
    )
    logger.info(f"Test loader creted (|test set|={len(data.test_idx)})")

    # Run evaluation
    # all_labels, all_preds = [], []
    all_node_ids, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            # print("Batch size:", batch.batch_size)
            # print("num_property_embedding:", batch.num_property_embedding)
            # print("Any NaNs?", torch.isnan(batch.num_property_embedding).any())
            # print("Min:", batch.num_property_embedding.min(), "Max:", batch.num_property_embedding.max())

            # print("cat_property_embedding:", batch.cat_property_embedding)
            # print("Any NaNs?", torch.isnan(batch.cat_property_embedding).any())
            # print("Min:", batch.cat_property_embedding.min(), "Max:", batch.cat_property_embedding.max())

            # print("num_for_h:", batch.num_for_h)
            # print("Any NaNs?", torch.isnan(batch.num_for_h).any())
            # print("Min:", batch.num_for_h.min(), "Max:", batch.num_for_h.max())

            batch = batch.to(device)
            batch.num_property_embedding = torch.nan_to_num(batch.num_property_embedding, nan=0.0)
            batch.num_for_h = torch.nan_to_num(batch.num_for_h, nan=0.0)
            out, _ = model(
                batch.des_embedding,
                batch.tweet_embedding,
                batch.num_property_embedding,
                batch.cat_property_embedding,
                batch.num_for_h,
                batch.edge_index,
                batch.edge_type,
                batch_size=batch.batch_size
            )
            n = batch.batch_size  # target nodes only
            preds = out[:n]
            node_ids = batch.user_id[:n]  # global node indices

            all_preds.append(preds.cpu())
            all_node_ids.append(node_ids.cpu())
            n = batch.batch_size
            # all_preds.append(out[:n])
            # all_labels.append(batch.y[:n])
    logger.info("Evaluation complete")

    # y, pred = torch.cat(all_labels), torch.cat(all_preds)
    # metrics, summary = calc_metrics(y, pred)
    # logger.info("Test results:" + summary)
    # print("Test results:" + summary)
    pred = torch.cat(all_preds)
    pred_np = pred.cpu().numpy()
    node_ids = torch.cat(all_node_ids)
    probs = torch.nn.functional.softmax(pred, dim=1)

    logits_np = pred.cpu().numpy().tolist()
    probs_np = probs.cpu().numpy().tolist()
    node_ids_np = node_ids.cpu().numpy()

    predictions = pred_np.argmax(axis=1)

    df = pd.DataFrame({
        'user_id': node_ids_np,
        'prediction': predictions,
        'logits': logits_np,
        'probabilities': probs_np
    })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")
    print(f"Predictions saved to {args.output}")

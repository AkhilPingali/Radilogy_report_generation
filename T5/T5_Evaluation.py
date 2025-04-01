import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    generated_reports = []
    true_reports = []

    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    total_bleu = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            image_features = batch['features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image_features=image_features, labels=labels)
            preds = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, ref in zip(preds, refs):
                # BLEU Score
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
                total_bleu += bleu

                # ROUGE Score
                rouge_scores = rouge.score(pred, ref)
                total_rouge1 += rouge_scores['rouge1'].fmeasure
                total_rouge2 += rouge_scores['rouge2'].fmeasure
                total_rougeL += rouge_scores['rougeL'].fmeasure

                generated_reports.append(pred)
                true_reports.append(ref)
                count += 1

    avg_bleu = total_bleu / count
    avg_rouge1 = total_rouge1 / count
    avg_rouge2 = total_rouge2 / count
    avg_rougeL = total_rougeL / count

    avg_metrics = {
    "BLEU": avg_bleu,
    "ROUGE-1": avg_rouge1,
    "ROUGE-2": avg_rouge2,
    "ROUGE-L": avg_rougeL,
}

    return generated_reports, true_reports,avg_metrics

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Tokenizer, AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast

from advanced_likhon_model import create_advanced_likhon35_model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    model = create_advanced_likhon35_model(
        vocab_size=args.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head
    )
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(args.dataset_name, split="train")
    val_dataset = load_dataset(args.dataset_name, split="validation")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

    train_sampler = DistributedSampler(tokenized_train, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(tokenized_train, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(tokenized_val, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler()

    if rank == 0:
        wandb.init(project="likhon-3.5", name=args.run_name, config=args)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            with autocast():
                _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * args.gradient_accumulation_steps

            if rank == 0 and step % args.log_interval == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": step
                })

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()

                _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if rank == 0:
            wandb.log({
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "epoch": epoch
            })
            
            # Save the model
            torch.save(model.module.state_dict(), f"{args.output_dir}/likhon35_epoch_{epoch}.pt")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Advanced Likhon 3.5 Model")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset to use for training")
    parser.add_argument("--run_name", type=str, default="likhon35_training", help="Name of the run for logging")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval for training steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for training")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size for the model")
    parser.add_argument("--n_embd", type=int, default=1024, help="Embedding dimension for the model")
    parser.add_argument("--n_layer", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=16, help="Number of attention heads")
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


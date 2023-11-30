def train(model, train_loader, optimizer, scheduler):
    model.train()
    total_loss=0
    with tqdm(train_loader, leave=True) as pbar:
        for step, (x,h,y) in enumerate(pbar):
            if h[0].item() is False:
                # no hour embedding
                x = x.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred,y)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item():.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )

                if scheduler is not None:
                    scheduler.step()
            else:
                # hour embedding
                x = x.to(torch.float32).to(device)
                h = h.long().to(device)
                y = y.to(torch.float32).to(device)
                optimizer.zero_grad()
                pred = model(x,h)
                loss = criterion(pred,y)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item():.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )

                if scheduler is not None:
                    scheduler.step()
        train_loss = total_loss/len(train_loader)
    return train_loss

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    val_loss=0
    with tqdm(val_loader, leave=True) as pbar:
        for x,h,y in pbar:
            if h[0].item() is False:
                x=x.to(torch.float32).to(device)
                y=y.to(torch.float32).to(device)
                pred = model(x)
                loss = criterion(pred,y)
                val_loss += loss.item()

                pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item():.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
            else:
                x=x.to(torch.float32).to(device)
                h=h.long().to(device)
                y=y.to(torch.float32).to(device)
                pred = model(x,h)
                loss = criterion(pred,y)
                val_loss += loss.item()

                pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item():.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
    val_loss /= len(val_loader)
    
    return val_loss


# Train folds !
def train_folds(model, train_dl, valid_dl, optimizer, scheduler, epochs, fold_num, save_name, save_dir):
    os.makedirs(f'./{save_dir}',exist_ok=True)
    
    history = {
        'train_loss': [],
        'valid_loss': [],
        'lr': [],
    }
    best_valid_loss = 1e5
    for epoch in range(epochs):
        
        train_loss = train(model, train_dl, optimizer, scheduler)
        valid_loss = evaluate(model, valid_dl)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join('./model/', f"model_best_fold-{fold_num}-{save_name}.pth")
            )
            print(f'[*] best valid with loss {valid_loss} model saved at ./model/ !')
        print(
            f"epoch{epoch+1} -- ",
            f"train_loss = {train_loss:.6f} -- ",
            f"valid_loss = {valid_loss:.6f}",
        )
    
    return history

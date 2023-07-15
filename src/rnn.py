import os
import copy
import time
import torch
import pandas as pd
import numpy as np

from src.utils import get_scaler, get_torch_optimizer, get_torch_criterion, load_fasttext_embeddings
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt


class ForecastModel(nn.Module):
    def __init__(self, model_type='lstm', input_size=4, num_layers=1, hidden_size=10, batch_first=True,
                 conditional_labels=None, emb_dict=None, dropout=0.3, output_size=1):
        super(ForecastModel, self).__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.conditional_labels = conditional_labels
        if self.conditional_labels is not None:
            if emb_dict is None:
                emb_dict = load_fasttext_embeddings(self.conditional_labels, self.hidden_size)
                embedding_weights = np.asarray([emb_dict[c] for c in self.conditional_labels])
                self.cond_emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())
            else:
                embedding_weights = PCA(n_components=self.hidden_size).fit_transform(
                    [emb_dict[c] for c in self.conditional_labels])
                self.cond_emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())

    def get_embeddings(self, cond_labels):
        embedding = self.cond_emb(
            torch.tensor([self.conditional_labels.index(x) for x in cond_labels], device=self.cond_emb.weight.device))
        hidden_state = embedding.unsqueeze(1).repeat(1, self.num_layers, 1).permute(1, 0, 2).contiguous()
        return hidden_state

    def forward(self, x, hidden_state=None):
        if self.model_type == 'lstm' and hidden_state is not None:
            hidden_state = (hidden_state, hidden_state)
        x, _ = self.rnn(x, hidden_state)
        if len(x.shape) > 2:
            x = x[:, -1, :]
        else:
            x = x[-1]
        x = self.linear(self.dropout(x))
        return x.squeeze(-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, x_dfs, y_arr, cond_col='Country'):  # , state_shape, initial_states=None):
        super(TimeSeriesDataset, self).__init__()
        assert all(cond_col in x_df.columns for x_df in x_dfs)
        assert len(x_dfs) == len(y_arr)
        self.x_dfs = x_dfs
        self.y_arr = y_arr
        self.cond_col = cond_col
        # self.state_shape = state_shape
        self.feat_col = [x for x in x_dfs[0].columns if x != cond_col]
        # if initial_states is None:
        #     initial_states = {}
        # self.initial_states = initial_states

    #     def initialize_states(self):
    #         for cls_label in np.unique([x[self.cond_col].unique()[0] for x in self.x_dfs]):
    #             self.get_state(cls_label)

    #     def get_initial_states(self):
    #         return {k: v.detach().clone() for k, v in self.initial_states.items()}

    #     def set_initial_states(self, initial_states):
    #         self.initial_states = initial_states

    def __len__(self):
        return len(self.x_dfs)

    # def get_state(self, cls_label):
    #     if cls_label in self.initial_states:
    #         return self.initial_states[cls_label].detach().clone()
    #     else:
    #         state = torch.rand(self.state_shape, dtype=torch.float)
    #         self.initial_states[cls_label] = state
    #         return state.detach().clone()

    def __getitem__(self, idx):
        x_row, y = self.x_dfs[idx], self.y_arr[idx]
        assert len(x_row[self.cond_col].unique()) == 1
        # initial_state = self.get_state()
        cond_label = x_row[self.cond_col].unique()[0]
        x_row = x_row[self.feat_col]
        return cond_label, x_row.to_numpy(), y


def load_data(imputed_dir, candidates, target, seq_len=5, freq='Y', test_size=4, remove_covid=False, reverse=False, country_filter=None,
              preprocess='standard', single_x_scaler=False, single_y_scaler=True, shift_target=True):
    seq_train_x, seq_train_y, seq_test_x, seq_test_y = [], [], [], []
    country_list = []
    y_scaler = {}
    all_df = []
    if isinstance(country_filter, str):
        country_filter = [country_filter]
    if shift_target and 'ShiftedTarget' not in candidates:
        candidates.append('ShiftedTarget')
    for f in os.listdir(imputed_dir):
        if f.endswith('.csv'):
            # print('processing', f)
            country = f.split('.')[0]
            if country_filter is not None:
                if country not in country_filter:
                    continue
            df = pd.read_csv(os.path.join(imputed_dir, f))
            df['Date'] = pd.to_datetime(df['Date'])
            if remove_covid:
                df = df[df['Date'] < '2020-01-01']
            if freq == 'Y':
                df = df.set_index('Date')
                df = df.resample('Y').first().reset_index(names='Date')

            if reverse:
                df = df.iloc[::-1]

            df['Country'] = country
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            if shift_target:
                df['ShiftedTarget'] = df[target].copy()

            train_df = df[[target] + candidates].dropna()
            if isinstance(test_size, float):
                test_idx = int(test_size * len(train_df))
            else:
                test_idx = test_size
            if not single_x_scaler:
                train_df, test_df = train_df.iloc[:-test_idx], train_df.iloc[-test_idx:]

                scaler = get_scaler(preprocess)  # scale each country stats separately

                temp = candidates.copy()
                temp.remove('Country')
                # temp.remove('Year')
                if shift_target:
                    temp.remove('ShiftedTarget')

                train_df[temp] = scaler.fit_transform(train_df[temp])
                test_df[temp] = scaler.transform(test_df[temp])
                train_df = pd.concat([train_df, test_df], ignore_index=True)
            if not single_y_scaler:
                train_df, test_df = train_df.iloc[:-test_idx], train_df.iloc[-test_idx:]

                scaler = get_scaler(preprocess)  # scale each country stats separately

                train_df[[target]] = scaler.fit_transform(train_df[[target]])
                test_df[[target]] = scaler.transform(test_df[[target]])
                train_df = pd.concat([train_df, test_df], ignore_index=True)
                y_scaler[country] = scaler
            all_df.append(train_df)
            for i in range(len(train_df) - seq_len):
                if i < len(train_df) - seq_len - test_idx:
                    seq_train_x.append(train_df[candidates].iloc[i: i + seq_len])
                    seq_train_y.append(train_df[target].iloc[i + seq_len])
                    country_list.append(f.split('.')[0])
                else:
                    seq_test_x.append(train_df[candidates].iloc[i: i + seq_len])
                    seq_test_y.append(train_df[target].iloc[i + seq_len])
    seq_train_y, seq_test_y = np.array(seq_train_y), np.array(seq_test_y)
    temp = pd.concat(all_df, ignore_index=True)
    x_scaler = get_scaler(preprocess)
    if single_x_scaler:
        temp_feat = candidates.copy()
        temp_feat.remove('Country')
        x_scaler.fit_transform(temp[temp_feat])
        for seq_df in seq_train_x:
            seq_df[temp_feat] = x_scaler.transform(seq_df[temp_feat])
        for seq_df in seq_test_x:
            seq_df[temp_feat] = x_scaler.transform(seq_df[temp_feat])
    else:
        x_scaler.fit_transform(temp[['ShiftedTarget']])
        for seq_df in seq_train_x:
            seq_df[['ShiftedTarget']] = x_scaler.transform(seq_df[['ShiftedTarget']])
        for seq_df in seq_test_x:
            seq_df[['ShiftedTarget']] = x_scaler.transform(seq_df[['ShiftedTarget']])

    if single_y_scaler:
        y_scaler = get_scaler(preprocess)
        seq_train_y = y_scaler.fit_transform(np.expand_dims(seq_train_y, -1)).flatten()
        seq_test_y = y_scaler.transform(np.expand_dims(seq_test_y, -1)).flatten()
    return seq_train_x, seq_train_y, seq_test_x, seq_test_y, x_scaler, y_scaler, country_list


def load_data_forecast(imputed_dir, candidates, target, seq_len=5, freq='Y', reverse=False, country_filter=None,
              preprocess='standard', single_x_scaler=False, single_y_scaler=True, shift_target=True):
    seq_train_x, seq_train_y, seq_test_x, seq_test_y = [], [], [], []
    country_list = []
    y_scaler = {}
    all_df = []
    if isinstance(country_filter, str):
        country_filter = [country_filter]
    if shift_target and 'ShiftedTarget' not in candidates:
        candidates.append('ShiftedTarget')
    for f in os.listdir(imputed_dir):
        if f.endswith('.csv'):
            # print('processing', f)
            country = f.split('.')[0]
            if country_filter is not None:
                if country not in country_filter:
                    continue

            df = pd.read_csv(os.path.join(imputed_dir, f))
            df['Date'] = pd.to_datetime(df['Date'])

            if freq == 'Y':
                df = df.set_index('Date')
                df = df.resample('Y').first().reset_index(names='Date')

            if reverse:
                df = df.iloc[::-1]

            df['Country'] = country
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            if shift_target:
                df['ShiftedTarget'] = df[target].shift()
            df = df[candidates + [target]].reset_index(drop=True)
            train_df = df.iloc[max([df[c].first_valid_index() for c in candidates + [target]]):].reset_index(drop=True)
            test_idx = train_df[target].last_valid_index() + 1

            if not single_x_scaler:
                train_df, test_df = train_df.iloc[:test_idx], train_df.iloc[test_idx:]

                scaler = get_scaler(preprocess)  # scale each country stats separately

                temp = candidates.copy()
                temp.remove('Country')
                # temp.remove('Year')
                if shift_target:
                    temp.remove('ShiftedTarget')

                train_df[temp] = scaler.fit_transform(train_df[temp])
                test_df[temp] = scaler.transform(test_df[temp])
                train_df = pd.concat([train_df, test_df], ignore_index=True)
            if not single_y_scaler:
                train_df, test_df = train_df.iloc[:test_idx], train_df.iloc[test_idx:]

                scaler = get_scaler(preprocess)  # scale each country stats separately

                train_df[[target]] = scaler.fit_transform(train_df[[target]])
                test_df[[target]] = scaler.transform(test_df[[target]])
                train_df = pd.concat([train_df, test_df], ignore_index=True)
                y_scaler[country] = scaler
            all_df.append(train_df)
            for i in range(len(train_df) - seq_len + 1):
                if i < test_idx + 1 - seq_len:
                    seq_train_x.append(train_df[candidates].iloc[i: i + seq_len])
                    seq_train_y.append(train_df[target].iloc[i + seq_len - 1])
                    country_list.append(f.split('.')[0])
                else:
                    seq_test_x.append(train_df[candidates].iloc[i: i + seq_len])
                    seq_test_y.append(train_df[target].iloc[i + seq_len - 1])
    seq_train_y, seq_test_y = np.array(seq_train_y), np.array(seq_test_y)
    temp = pd.concat(all_df, ignore_index=True)
    x_scaler = get_scaler(preprocess)
    if single_x_scaler:
        temp_feat = candidates.copy()
        temp_feat.remove('Country')
        x_scaler.fit_transform(temp[temp_feat])
        for seq_df in seq_train_x:
            seq_df[temp_feat] = x_scaler.transform(seq_df[temp_feat])
        for seq_df in seq_test_x:
            seq_df[temp_feat] = x_scaler.transform(seq_df[temp_feat])
    else:
        x_scaler.fit_transform(temp[['ShiftedTarget']])
        for seq_df in seq_train_x:
            seq_df[['ShiftedTarget']] = x_scaler.transform(seq_df[['ShiftedTarget']])
        for seq_df in seq_test_x:
            seq_df[['ShiftedTarget']] = x_scaler.transform(seq_df[['ShiftedTarget']])

    if single_y_scaler:
        y_scaler = get_scaler(preprocess)
        seq_train_y = y_scaler.fit_transform(np.expand_dims(seq_train_y, -1)).flatten()
        seq_test_y = y_scaler.transform(np.expand_dims(seq_test_y, -1)).flatten()
    return seq_train_x, seq_train_y, seq_test_x, seq_test_y, x_scaler, y_scaler, country_list


def train_rnn(model, criterion, optimizer, train_dataloader, val_dataloader=None, epochs=50, early_stop_delta=1e-4,
              early_stop_patience=3, verbose=True, cuda=True, save_best=True):
    if cuda:
        model.cuda()
    best_val_loss = 100000
    counter = 0
    train_losses, val_losses = [], []
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        st = time.time()
        for cond_labels, x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.float(), y_batch.float()
            init_state = model.get_embeddings(cond_labels)
            if cuda:
                init_state, x_batch, y_batch = init_state.cuda(), x_batch.cuda(), y_batch.cuda()
            # init_state = init_state.permute(1, 0, 2).contiguous() # switch batch to middle for pytorch
            y_pred = model(x_batch, init_state)
            loss = criterion(y_pred, y_batch)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        if val_dataloader is not None:
            _, val_loss = eval_rnn(model, criterion, val_dataloader, cuda=cuda)
            if verbose:
                print("Epoch %d: train loss %.4f, val loss %.4f, time %.4f" % (
                epoch, train_loss, val_loss, time.time() - st))
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0
            elif val_loss > best_val_loss + early_stop_delta and epoch > 10:
                counter += 1
                if counter >= early_stop_patience:
                    if verbose:
                        print("Early stopping as no improvement in val loss")
                    break
        else:
            if verbose:
                print("Epoch %d: train loss %.4f, time %.4f" % (epoch, train_loss, time.time() - st))
    if epoch == epochs - 1 and verbose:
        print('warning not converged')
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


@torch.no_grad()
def eval_rnn(model, criterion, dataloader, cuda=True):
    val_loss = 0.
    preds = []
    if cuda:
        model.cuda()
    model.eval()
    for cond_labels, x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.float(), y_batch.float()
        init_state = model.get_embeddings(cond_labels)
        if cuda:
            init_state, x_batch, y_batch = init_state.cuda(), x_batch.cuda(), y_batch.cuda()
        # init_state = init_state.permute(1, 0, 2).contiguous() # switch batch to middle for pytorch
        y_pred = model(x_batch, init_state)
        loss = criterion(y_pred, y_batch)
        val_loss += loss.item()
        preds.append(y_pred.detach().cpu().numpy())
    return np.concatenate(preds), val_loss / len(dataloader)


@torch.no_grad()
def one_step_prediction(model, seq_x, x_scaler, y_scaler, cond_col='Country', cuda=True, shift_target=True):
    # helper function to perform one step forecast sequentially
    # seq_x is list of dataframes in sequential order grouped by country
    final_predictions = {}
    if cuda:
        model.cuda()
    model.eval()
    for x_df in seq_x:
        x_df = x_df.copy()
        assert len(x_df[cond_col].unique()) == 1
        cond_label = x_df[cond_col].unique()[0]
        if shift_target:
            nan_count = x_df['ShiftedTarget'].isna().sum()
            if nan_count > 0:
                assert nan_count <= len(final_predictions[cond_label])
                shift_fill = x_scaler.transform(pd.DataFrame({'ShiftedTarget': final_predictions[cond_label][-nan_count:]})).flatten()
                x_df.iloc[-nan_count:, x_df.columns.get_loc('ShiftedTarget')] = shift_fill
            assert x_df['ShiftedTarget'].isna().sum() == 0
        init_state = model.get_embeddings([cond_label])
        x_input = torch.from_numpy(x_df[[col for col in x_df.columns if col != cond_col]].to_numpy()).float().unsqueeze(0)
        if cuda:
            init_state, x_input = init_state.cuda(), x_input.cuda()
        pred = model(x_input, init_state).detach().cpu().item()
        pred = y_scaler.inverse_transform([[pred]])[0][0]
        if cond_label in final_predictions:
            final_predictions[cond_label].append(pred)
        else:
            final_predictions[cond_label] = [pred]
    return final_predictions


def grid_search_rnn(imputed_dir, candidates, target, emb_dict=None, freq='Y', test_size=4, reverse=False, preprocess=None, country_filter=None, criterion_type='mse', model_types=None,
                    param_grids=None, single_x_scaler=False, single_y_scaler=True, max_epochs=100, verbose=True, cuda=True, input_size=4):
    if preprocess is None:
        preprocess = ['standard']
    if param_grids is None:
        param_grids = {}
    if model_types is None:
        model_types = ['lstm', 'gru']
    criterion = get_torch_criterion(criterion_type)

    best_score = np.inf
    best_param = None
    for seq_len in param_grids.get('seq_len', [5, 7]):
        for prep in preprocess:
            seq_train_x_ori, seq_train_y_ori, _, _, _, _, train_country = load_data(imputed_dir, candidates, target, seq_len, freq=freq, test_size=test_size, reverse=reverse, preprocess=prep, country_filter=country_filter,
                                                                                    single_x_scaler=single_x_scaler, single_y_scaler=single_y_scaler)
            for batch_size in param_grids.get('batch_size', [16, 32]):
                for model_type in model_types:
                    for num_layers in param_grids.get('num_layers', range(1, 2)): # 2 layers result in worse performance
                        for hidden_size in param_grids.get('hidden_size', [10, 15, 20]):
                            for optim_type in param_grids.get('optim_type', ['sgd', 'nadam', 'rmsprop']):
                                for lr in [0.001, 0.0005]:
                                    model = ForecastModel('gru', input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, conditional_labels=np.unique(train_country).tolist(), emb_dict=emb_dict)
                                    optimizer = get_torch_optimizer(optim_type, model.parameters(), lr=lr)
                                    score_list = []
                                    kf = StratifiedKFold()
                                    for train_index, test_index in kf.split(seq_train_x_ori, train_country):
                                        seq_train_x, seq_val_x = [seq_train_x_ori[i] for i in train_index], [seq_train_x_ori[i] for i in test_index]
                                        seq_train_y, seq_val_y  = seq_train_y_ori[train_index], seq_train_y_ori[test_index]
                                        train_dataset = TimeSeriesDataset(seq_train_x, seq_train_y,)# (model.num_layers, model.hidden_size))
                                        # train_dataset.initialize_states()
                                        val_dataset = TimeSeriesDataset(seq_val_x, seq_val_y,)# (model.num_layers, model.hidden_size))
                                        # val_dataset.set_initial_states(train_dataset.get_initial_states())
                                        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
                                        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, pin_memory=True)
                                        model, train_losses, val_losses = train_rnn(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=max_epochs, verbose=False, cuda=cuda)

                                        score_list.append(sqrt(val_losses[-1]) if criterion_type == 'mse' else val_losses[-1])
                                    score = np.mean(score_list)
                                    if score < best_score:
                                        best_score = score
                                        best_param = ({'preprocess': prep, 'seq_len': seq_len, 'batch_size': batch_size}, {'model_type': model_type, 'num_layers': num_layers, 'hidden_size': hidden_size},
                                                      {'optim_type': optim_type, 'lr': lr})
                                    if verbose:
                                        print(({'preprocess': prep, 'seq_len': seq_len, 'batch_size': batch_size}, {'model_type': model_type, 'num_layers': num_layers, 'hidden_size': hidden_size},
                                              {'optim_type': optim_type, 'lr': lr}), round(score, 5), round(np.std(score_list), 5))
    return best_param, best_score

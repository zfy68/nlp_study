'''
Encoder-Decoder 模型构建
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import jieba
from torch.utils.tensorboard import SummaryWriter

# 注意力机制Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # hidden_dim * 4是因为encoder_outputs和hidden都是hidden_dim*2
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, hidden_dim*2]
        encoder_outputs: [batch_size, seq_len, hidden_dim*2]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim*2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, seq_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        return F.softmax(attention, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout, return_method='concat'):
        super(Encoder, self).__init__()
        self.method = return_method
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        
        # 分开前向和后向
        if self.method == 'concat':
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], dim=2)
        else:  # 'add'
            hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]

        return outputs, hidden

# 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, return_method='concat'):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_method = return_method

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_dim)

        decoder_hidden_dim = hidden_dim * 2 if return_method == 'concat' else hidden_dim

        self.gru = nn.GRU(emb_dim + hidden_dim * 2, decoder_hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(decoder_hidden_dim + hidden_dim * 2 + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        attn_hidden = hidden[-1]  # [batch_size, hidden_dim*2]（因为concat了）

        attention_weights = self.attention(attn_hidden, encoder_outputs)  # [batch_size, 1, seq_len]
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, hidden_dim*2]

        gru_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hidden_dim*2]

        output, hidden = self.gru(gru_input, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)

        output = torch.cat((output, context, embedded), dim=1)  # [batch_size, decoder_hidden_dim + hidden_dim*2 + emb_dim]
        prediction = self.fc_out(output)  # [batch_size, output_dim]

        return prediction, hidden, attention_weights

# Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, max_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs
    
# 训练模型
def train(model, data_loader, optimizer, criterion, device, writer, epoch):
    '''
    训练模型
    model: 模型
    data_loader: 数据加载器
    optimizer: 优化器
    criterion: 损失函数
    device: 设备
    writer: TensorBoard记录器
    epoch: 训练轮数
    '''
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(data_loader):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(src, trg)
        # 计算损失
        # 计算损失
        output_dim = output.shape[-1]
        # 忽略第一个时间步(BOS)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1) 
        # 计算损失
        loss = criterion(output, trg)
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # 更新参数
        optimizer.step()
        epoch_loss += loss.item()
        # 每50个批次记录损失
        if (i + 1) % 50 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}/{len(data_loader)}, Loss: {loss.item():.4f}')
            writer.add_scalar('Training Loss/Batch', loss.item(), epoch * len(data_loader) + i)
    return epoch_loss / len(data_loader)

# 评估模型
def evaluate(model, data_loader, criterion, device):
    '''
    评估模型
    model: 模型
    data_loader: 数据加载器
    criterion: 损失函数
    device: 设备
    '''
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            # 前向传播
            output = model(src, trg, teacher_forcing_ratio=0)
            # 计算损失
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1) 
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# 生成对联函数
def generate(model, src, max_len, device):
    '''
    生成对联
    model: 模型
    src: 输入序列
    max_len: 最大长度
    device: 设备
    '''
    model.eval()
    src = src.to(device)
    batch_size = src.shape[0]
    trg_vocab_size = model.decoder.output_dim
    outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        input = torch.tensor([1] * batch_size).to(device)  # BOS token
        for t in range(max_len):
            output, hidden, _ = model.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            input = output.argmax(1)  # greedy decoding
    return outputs

def load_data(data_dir, max_length):
    '''
    加载数据
    vocab_path: 词汇表路径
    max_length: 最大长度
    '''
    # 读取词汇表
    vocab_path = os.path.join(data_dir, 'vocabs')
    with open (vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    # 添加特殊标记
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']        
    vocab = special_tokens + [word for word in vocab if word not in special_tokens]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)
    # 读取数据
    train_in_path = os.path.join(data_dir,'train', 'in.txt')
    train_out_path = os.path.join(data_dir, 'train', 'out.txt')
    train_src,train_tgt = load_paired_data(train_in_path, train_out_path)

    test_in_path = os.path.join(data_dir, 'test', 'in.txt')
    test_out_path = os.path.join(data_dir, 'test', 'out.txt')
    test_src,test_tgt = load_paired_data(test_in_path, test_out_path)


    # 预处理数据
    train_src_data,train_tgt_data =  preprocess_data(train_src, train_tgt, word2idx, word2idx)
    test_src_data,test_tgt_data =  preprocess_data(test_src, test_tgt, word2idx, word2idx)
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(train_src_data, train_tgt_data)
    test_dataset = torch.utils.data.TensorDataset(test_src_data, test_tgt_data)
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, word2idx, idx2word, vocab_size


def preprocess_data(src_sentences, tgt_sentences, word2idx, idx2word,max_len=30):
    '''文本数据转为索引'''
    src_data = []
    tgt_data = []

    for src,tgt in zip(src_sentences,tgt_sentences):
        # jieba分词
        src_words = list(jieba.cut(src))
        tgt_words = list(jieba.cut(tgt))
        # 转索引
        src_idx =[word2idx.get(word, word2idx['<unk>']) for word in src_words]
        tgt_idx = [word2idx.get(word, word2idx['<unk>']) for word in tgt_words]

        # 添加起始和结束标记
        src_idx = [word2idx['<sos>']] + src_idx + [word2idx['<eos>']]
        tgt_idx = [word2idx['<sos>']] + tgt_idx + [word2idx['<eos>']]

        # 填充或截断
        src_idx = src_idx[:max_len] + [word2idx['<pad>']] * (max_len - len(src_idx))
        tgt_idx = tgt_idx[:max_len] + [word2idx['<pad>']] * (max_len - len(tgt_idx))
        
        src_idx = src_idx[:max_len]
        tgt_idx = tgt_idx[:max_len]
        # 添加到数据集中
        src_data.append(src_idx)
        tgt_data.append(tgt_idx)
    
    # 转为tensor
    return torch.tensor(src_data), torch.tensor(tgt_data)


def load_paired_data(in_path,out_path):
    '''
    加载对联数据
    '''
    with open(in_path, 'r', encoding='utf-8') as f_in, open(out_path, 'r', encoding='utf-8') as f_out:
        inputs = [line.strip() for line in f_in.readlines()]
        outputs = [line.strip() for line in f_out.readlines()]

    assert len(inputs) == len(outputs), "输入和输出数据长度不一致"
    return inputs, outputs
      


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print('加载数据...')
    train_loader, test_loader,word2idx,idx2word,vocab_size= load_data('/kaggle/input/chinese-couplets/couplet/',50)  # 假设load_data函数已经定义
    print('数据加载完成...')

    # 超参数
    input_dim = vocab_size
    emb_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.5
    output_dim = vocab_size

    # 定义模型
    print("创建模型...")
    encoder = Encoder(input_dim, emb_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hidden_dim, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型总参数数量: {total_params:,}')


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>']) # 忽略填充标记
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) 
    # TensorBoard
    writer = SummaryWriter(log_dir='/kaggle/working/runs')
    # 训练模型
    print("开始训练...")
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, writer, epoch)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)

        # 保存最佳模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'/kaggle/working/model_epoch_{epoch+1}.pth')
            print(f'模型已保存: model_epoch_{epoch+1}.pth')
        
    # 根据模型测试一些样例
    model.load_state_dict(torch.load('/kaggle/working/runs/model_epoch_10.pth'))
    test_input = ["春眠不觉晓", "夜来风雨声"]
    test_input = [torch.tensor([word2idx.get(word, word2idx['<unk>']) for word in list(jieba.cut(sentence))]) for sentence in test_input]
    test_input = torch.stack(test_input).to(device)
    max_len = 50
    generated_output = generate(model, test_input, max_len, device)
    generated_output = generated_output.argmax(2)
    # 将索引转换为单词
    generated_output = generated_output.cpu().numpy()
    generated_output = [[idx2word[idx] for idx in sentence if idx != word2idx['<pad>']] for sentence in generated_output]
    # 打印生成的对联
    for i, sentence in enumerate(generated_output):
        print(f"输入: {test_input[i]}, 生成的对联: {' '.join(sentence)}")
    

if __name__ == '__main__':
    main()
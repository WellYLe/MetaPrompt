import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 获取项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from prompt_graph.data.load4data import load4link_prediction_multi_graph, load4link_prediction_single_graph, load4link_prediction_single_graph_modified
from torch.optim import Adam
import time

from prompt_graph.defines import GRAPH_TASKS, NODE_TASKS
from prompt_graph.pretrain.base import PreTrain


class Edgepred_GPPT(PreTrain):
    def __init__(self, *args, datapath=None, **kwargs):    
        super().__init__(*args, **kwargs)
        # 如果datapath是相对路径且不是绝对路径，则将其转换为绝对路径
        if datapath and not os.path.isabs(datapath):
            # 检查是否已经包含了项目路径前缀
            if datapath.startswith('prompt_graph/'):
                # 获取项目根目录
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.datapath = os.path.join(project_root, datapath)
            else:
                # 获取脚本所在目录
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.datapath = os.path.join(script_dir, datapath)
        else:
            self.datapath = datapath
            
        self.dataloader = self.generate_loader_data()
        self.initialize_gnn(self.input_dim, self.hid_dim) 
        self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.hid_dim).to(self.device)  # output_dim 未出现


    def generate_loader_data(self):
        if self.dataset_name in NODE_TASKS:
            if self.datapath is None:
                self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(self.dataset_name)  
            else:
                self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph_modified(self.datapath)  
            #现在输入的是扰动图了
            self.data.to(self.device) 
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            if self.dataset_name in['ogbn-arxiv', 'Flickr']:#单图数据集
                return DataLoader(data, batch_size = 1024, shuffle=True, num_workers=self.num_workers)
            else:
                return DataLoader(data, batch_size=64, shuffle=True, num_workers=self.num_workers)
        
        elif self.dataset_name in GRAPH_TASKS:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(self.dataset_name)
            self.data.to(self.device)
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            
            # Batch图太大，向前传播的时候分开操作
            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:#多图数据集
                self.batch_dataloader = DataLoader(self.data.to_data_list(),batch_size=256,shuffle=False, num_workers=self.num_workers)
                return DataLoader(data, batch_size=512000, shuffle=True, num_workers=self.num_workers)

            return DataLoader(data, batch_size=64, shuffle=True, num_workers=self.num_workers)
      
    def pretrain_one_epoch(self):

        accum_loss, total_step = 0, 0
        device = self.device

        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.gnn.train()
        all_preds = []
        all_labels = []
        
        for step, (batch_edge_label, batch_edge_index) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            batch_edge_label = batch_edge_label.to(device)
            batch_edge_index = batch_edge_index.to(device)

            # 如果graph datasets经过Batch图太大了，那就分开操作
            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:#多图数据集
                for batch_id, batch_graph in enumerate(self.batch_dataloader):
                    batch_graph.to(device)#将每个batch图放到device上
                    if(batch_id==0):
                        out = self.gnn(batch_graph.x, batch_graph.edge_index)#将每个batch图的节点特征和边索引放到gnn中进行前向传播
                    else:
                        out = torch.concatenate([out, self.gnn(batch_graph.x, batch_graph.edge_index)],dim=0)
            else:
                out = self.gnn(self.data.x, self.data.edge_index)#将整个数据集的节点特征和边索引放到gnn中进行前向传播   
                
            
            node_emb = self.graph_pred_linear(out)#将gnn的输出放到全连接层中进行前向传播，得到节点嵌入（node embedding）
          
            batch_edge_index = batch_edge_index.transpose(0,1)#将batch_edge_index的维度从(2, num_edges)转置为(num_edges, 2)，方便后续的解码操作
            batch_pred_log = self.gnn.decode(node_emb,batch_edge_index).view(-1)#将节点嵌入（node embedding）和边索引（batch_edge_index）放到解码层中进行前向传播，得到边预测的logits（batch_pred_log）
            loss = criterion(batch_pred_log, batch_edge_label)#计算二分类损失（loss）   
            #如果我训练图是污染的，训练时候觉得污染的是对的，是不是就意味着他在下游也会选择本不应该存在的边？
            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1
            
            # 收集预测结果和标签，用于计算指标
            with torch.no_grad():
                batch_pred_prob = torch.sigmoid(batch_pred_log)
                all_preds.append(batch_pred_prob.detach().cpu())
                all_labels.append(batch_edge_label.detach().cpu())
            
            print('第{}次反向传播过程'.format(step))

        # 计算整个epoch的指标
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.calculate_metrics(all_preds, all_labels)
        
        return accum_loss / total_step, metrics
        
    def calculate_metrics(self, preds, labels):
        """计算各种评估指标"""
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
        
        # 转换为numpy数组
        preds_np = preds.numpy()
        labels_np = labels.numpy()
        
        # 计算AUC
        auc = roc_auc_score(labels_np, preds_np)
        
        # 计算AP (Average Precision)
        ap = average_precision_score(labels_np, preds_np)
        
        # 计算准确率 (需要将概率转换为二分类预测)
        pred_labels = (preds_np > 0.5).astype(int)
        acc = accuracy_score(labels_np, pred_labels)
        
        return {
            'auc': auc,
            'ap': ap,
            'accuracy': acc
        }

    def pretrain(self):
        num_epoch = self.epochs
        train_loss_min = 1000000
        patience = 10
        cnt_wait = 0
                 
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss, metrics = self.pretrain_one_epoch()
            print(f"Edgepred_GPPT [Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"AUC {metrics['auc']:.4f} | AP {metrics['ap']:.4f} | Accuracy {metrics['accuracy']:.4f} | "
                  f"Cost Time {time.time() - st_time:.3f}s")
            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break
            print(cnt_wait)
        folder_path = f"./Experiment/pre_trained_model_self/{self.dataset_name}+'2' "
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        torch.save(self.gnn.state_dict(),
                    "{}/{}.{}.{}.pth".format(folder_path,'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
        
        print("+++model saved ! {}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))


import torch
import pdb
from kazuto_main_gc import KazutoMain
import pandas as pd

class Accuracy:
    def per_class_accuracy(self, pred_stack, target_stack, calculate_per_class_acc, calculate_overall_acc, num_classes, resume_model_path, save_path):
        # pdb.set_trace()
        correct_predicted_labels = torch.zeros(num_classes,dtype=torch.float64)
        total_labels = torch.zeros(num_classes,dtype=torch.float64)

        correct = pred_stack.eq(target_stack)
        
        res = []
        # topk=(1, 5)
        topk=(5,)
        for k in topk:
            if calculate_per_class_acc:
                for i in range(num_classes):
                    indices_of_occurance_of_i = target_stack[:k]==i
                    
                    correct_k = correct[:k]
                    if k==1:
                        num_occurance_of_i = torch.sum(indices_of_occurance_of_i)
                        indices_of_occurance_of_i = indices_of_occurance_of_i.squeeze()
                        correct_k = correct_k.reshape(-1)
                    else:
                        num_occurance_of_i = torch.sum(indices_of_occurance_of_i[0]) #counting the number of occurance using one row only
                    correct_predicted_labels[i] += correct_k[indices_of_occurance_of_i].float().sum(0, keepdim=True).item()
                    total_labels[i] += num_occurance_of_i
                
                per_class_accuracy = correct_predicted_labels/total_labels
                print('per_class_top'+str(k)+'_accuracy_epoch'+resume_model_path.split("_")[3]+":",per_class_accuracy)
                torch.save(per_class_accuracy, save_path+'per_class_top'+str(k)+'_accuracy_epoch'+resume_model_path.split("_")[3])
            
            if calculate_overall_acc:
                overall_correct_preds = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                overall_accuracy = overall_correct_preds/correct[:k].shape[1]
                res.append(overall_accuracy.item())
        if calculate_overall_acc:
            torch.save(res, save_path+'overall/top'+str(topk[0])+'_top'+str(topk[1])+'_accuracy_epoch'+resume_model_path.split("_")[3])

    def get_accuracy_df(self, path, filename):
        kazuto = KazutoMain()
        classes = kazuto.get_classtable()
        is_first = True
        for i in range(1,91):
            x=torch.load(path+filename.format(i))
            if is_first:
                stack = x
                is_first = False
            else:
                stack = torch.vstack((stack,x))

        accuracy_stack = stack.numpy()
        px2 = pd.DataFrame(accuracy_stack)

        px2.columns = classes
        return px2
    
    def get_stable_classes_after_epoch_number(self, accuracy_threshold, df):
        # eg accuracy_threshold = 0.9
        column_names_dict = {}
        for i in range(10,90,10):
            sub_df = df.iloc[i:]
            filter = (sub_df>=accuracy_threshold).all()
            column_names = sub_df.loc[:,filter].columns
            column_names_dict[i] = column_names.tolist()
        
        print("Between 10 and 20",  set(column_names_dict[10]))
        for i in range(10,80,10):
            print("Between", i+10, "and", (i+20),  set(column_names_dict[i+10])-set(column_names_dict[i]))

        
文件结构：

HW1P1                           // 已包含bonus作业（adam, adamW, dropout）
├─ mytorch                      
│  ├─ models                    
│  │  ├─ __init__.py            
│  │  └─ mlp.py                 
│  ├─ nn                        
│  │  ├─ __init__.py            
│  │  ├─ activation.py          
│  │  ├─ batchnorm.py           
│  │  ├─ dropout.py             
│  │  ├─ linear.py              
│  │  └─ loss.py                
│  ├─ optim                     
│  │  ├─ __init__.py            
│  │  ├─ adam.py                
│  │  ├─ adamW.py               
│  │  └─ sgd.py                 
│  └─ __init__.py               
├─ test_adam_dropout            // 验证文件
│  ├─ adamW_sol_W.pkl           
│  ├─ adamW_sol_b.pkl           
│  ├─ adam_sol_W.pkl            
│  ├─ adam_sol_b.pkl            
│  ├─ dropout_sol_backward.pkl  
│  └─ dropout_sol_forward.pkl   
├─ HW1P1_S24_Writeup.pdf        
├─ S24_HW1_Bonus.pdf            //验证bonus作业代码
├─ bonus_autograder.py          
├─ hw1p1_autograder.py          //验证HW1P1作业代码
├─ hw1p1_autograder_flags.py    
├─ readme.md   
├─ HW1P1.pdf                    // 个人笔记
└─ requirements.txt             

验证HW1P1作业代码：python hw1p1_autograder.py
验证bonus作业代码：python bonus_autograder.py


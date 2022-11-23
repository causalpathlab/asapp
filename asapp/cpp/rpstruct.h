
#ifndef BINARYTREE_H
#define BINARYTREE_H
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include<algorithm>
using namespace std;

struct Node
{
    int level;
    std::vector<int>  *pos_indxs_vec;
    std::vector<int>  *neg_indxs_vec;
    Node *parent;
    Node *pos_child;
    Node *neg_child;
};

class BinaryTree
{
    private:

        Node *root;

    public:

        BinaryTree()
        {
            root = nullptr;
        }

        ~BinaryTree(){

        }

        void insert_root(Node *&);

};

void BinaryTree::insert_root(Node * &node_ptr) {
    root = node_ptr;
}


void random_project(int n,std::vector<double> *rp_ptr){

    std::random_device rd{};
    std::mt19937 gen{rd()};

    normal_distribution<double> distN(0,1); 
    
    for (int i=1; i<=n; i++)
    {
      rp_ptr->push_back(distN(gen)); 
    }   

}

void split( Eigen::MatrixXd mat_A )
{   
int rows = mat_A.rows();
int cols = mat_A.cols();

int n = cols;
std::vector<double> rp_vals_vec;
random_project(n,&rp_vals_vec); 
Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(rp_vals_vec.data(), rp_vals_vec.size());
Eigen::VectorXd res = mat_A * b;


Node current_node;
int l = 0;
std::vector<int> pi;
std::vector<int> ni;
current_node.level = l;
current_node.pos_indxs_vec = &pi;
current_node.neg_indxs_vec = &ni;
current_node.parent = nullptr;
current_node.pos_child = nullptr;
current_node.neg_child = nullptr;

for ( Eigen::VectorXd::Index j = 0; j < res.size(); ++j)    
{
    if(res(j)>0){
    current_node.pos_indxs_vec->push_back(j);
}else{
    current_node.neg_indxs_vec->push_back(j);

}
}

std::vector<int> current_indexes;
for(std::vector<int>::iterator it = current_node.pos_indxs_vec->begin(); it != current_node.pos_indxs_vec->end(); ++it) {
    // std::cout<<*it<<std::endl;
    current_indexes.push_back(*it);
    
 }

// int test = mat_A(Eigen::seq(2,5)).rows();

}

#endif

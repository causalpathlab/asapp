#include<iostream>
#include<list>

using namespace std;


typedef struct 
{
    int cid;
    int dist;
}node;

typedef struct 
{
    int pid;
    std::list<node> child_nodes;
}pnode;

class Graph {

    private:

        int n_pnodes;
        list<pnode> *parent_nodes;

    public:

        Graph() {
        }

        Graph (int nodes) { 
            parent_nodes = new list<pnode> [nodes];
            this->n_pnodes = nodes;
        }

        ~Graph () { 
            delete [] parent_nodes;
        }

        void addParent(pnode p){
            parent_nodes->push_back(p);
        }

        void iterate(int idx){
            for(auto & p:parent_nodes[idx]){
                cout <<p.pid<<endl;
                for(auto & c:p.child_nodes){
                    cout<<c.cid<<endl;
                }
            }
        }

};

pnode update_graph(list<int> nbrs){


    int parent = nbrs.front();nbrs.remove(parent);
    int nbr_size = nbrs.size() - 1;


    pnode n_parent;
    n_parent.pid = parent;
    list<node> cn;
     
    int dist = 0;
    for(auto i : nbrs){

        node n_child;
        n_child.cid = i;
        n_child.dist = dist + 1;
        cn.push_front(n_child);
    }

    n_parent.child_nodes = cn;

    return n_parent;
}

int main()
{
    Graph g(2);

    list<int> nbrs{1,2,3,4};
    g.addParent(update_graph(nbrs));
    list<int> nbrs2{3,2,4,5};
    g.addParent(update_graph(nbrs2));

    cout << "Adjacency list implementation for graph" << endl;

    g.iterate(0);
    g.iterate(1);
}
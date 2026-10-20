#include "global_tp_layout_r1261.hpp"
#include <algorithm>
#include <climits>
#include <iostream>
#include <set>
#include <vector>
using infinilm::parallel::CrossNodeTpLayoutR1261;
void require(bool v){if(!v)throw std::runtime_error("1261 layout contract");}
int main(){
    size_t layouts=0, ranks=0, invalid=0;
    for(int local:{1,2,8})for(int nodes:{1,2,4})for(int pp:{1,2,3,4}) {
        std::set<int> world, participants;
        for(int stage=0;stage<pp;++stage){std::set<int> tp;
            for(int node=0;node<nodes;++node){CrossNodeTpLayoutR1261 x(local,nodes,node,pp,stage);
                require(x.local_size()==local && x.tp_size()==local*nodes && x.world_size()==local*nodes*pp);
                participants.insert(x.participant_rank());
                for(int rank=0;rank<local;++rank){int t=x.tp_rank(rank),w=x.world_rank(rank);tp.insert(t);world.insert(w);++ranks;
                    require(w==stage*x.tp_size()+t);
                    if(stage)require(w-x.tp_size()==(stage-1)*local*nodes+t);
                    if(stage+1<pp)require(w+x.tp_size()==(stage+1)*local*nodes+t);
                }
                for(int bad:{-1,local}){bool rejected=false;try{x.tp_rank(bad);}catch(const std::out_of_range&){rejected=true;}require(rejected);++invalid;}
            }
            require(tp.size()==size_t(local*nodes)&&*tp.begin()==0&&*tp.rbegin()==local*nodes-1);
        }
        require(world.size()==size_t(local*nodes*pp)&&*world.begin()==0&&*world.rbegin()==local*nodes*pp-1);
        require(participants.size()==size_t(nodes*pp)&&*participants.begin()==0&&*participants.rbegin()==nodes*pp-1);++layouts;
    }
    for(auto a:std::vector<std::vector<long long>>{{0,1,0,1,0},{8,0,0,1,0},{8,2,-1,2,0},{8,2,2,2,0},{8,2,0,0,0},{8,2,0,2,-1},{8,2,0,2,2},{INT_MAX,2,0,1,0},{8,2,0,INT_MAX,0}}){bool rejected=false;
        try{CrossNodeTpLayoutR1261 x{size_t(a[0]),size_t(a[1]),int(a[2]),int(a[3]),int(a[4])};}catch(const std::invalid_argument&){rejected=true;}
        require(rejected);++invalid;
    }
    std::cout<<"{\"completed\":true,\"layouts\":"<<layouts<<",\"ranks\":"<<ranks<<",\"invalid_cases\":"<<invalid<<",\"scope\":\"1261 value-only local/global rank coverage, PP neighbor ranks, participants and overflow; not GPU/model proof\"}"<<std::endl;
}

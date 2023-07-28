#ifndef BOIDS_H
#define BOIDS_H
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Sparse>
#include <math.h>
template <typename T, int dim>
using Vector = Eigen::Matrix<T, dim, 1, 0, dim, 1>;
template <typename T, int n, int m>
using Matrix = Eigen::Matrix<T, n, m, 0, n, m>;

// add more for yours
enum MethodTypes
{
    FREEFALL = 0,
    CIRCULAR = 1,
    SEPARATION = 2,
    ALIGNMENT = 3,
    COHESION = 4,
    LEADER = 5,
    COLLABORATIVEANDADVERSARIAL = 6
    
};
enum IntegrationMethodTypes
{
    BasicIntergration = 0,
    SymplecticEuler = 1,
    Explicitmidpoint = 2

};

template <class T, int dim>
class Boids
{
    typedef Matrix<T, Eigen::Dynamic, 1> VectorXT;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack; //a vector of type T with dimension "dim", which stands for a stack of TV
    typedef Vector<T, dim> TV;
    typedef Matrix<T, dim, dim> TM;

private:
    TVStack positions, positionstemp;
    TVStack velocities, velocitiestemp;
    int n;
    bool update = false;

public:
    Boids() : n(1) {} //?
    Boids(int n) : n(n)
    {
        initializePositions();
    }
    ~Boids() {}

    void setParticleNumber(int n) { n = n; }
    int getParticleNumber() { return n; }
    void initializePositions()
    {
        positions = TVStack::Zero(dim, n).unaryExpr([&](T dummy) { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX); }); 
        velocities = TVStack::Zero(dim, n).unaryExpr([&](T dummy) { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX); });
    }

    //TM Mass = TM::Identity();
    TV mouseposition;
    T h_step = 0.01;
    T g = 9.8;
    T threshold = 0.3;
    T threshold_collision = 10;
    //T expecteddistance = 20;
    T scale = 0.3;
    TV obstaclecenter = TV(360,360);//define the collision center in the screen coordinate
    TV obstaclecenter_sim = coordinatetranslation(obstaclecenter, scale);//(0.5,0.5)define the collision center in the point generation coordinate
    T obstacleradius = 40;
    T obstacleradius_sim = distance(obstaclecenter_sim,coordinatetranslation(obstaclecenter+TV(0,obstacleradius),scale));//5/27=0.185
    tt
    
    void getmouseposition(TV pos)
    {
        mouseposition = pos;
    }

    void getslider(T step)
    {
        h_step = step;
    }

    //translate the coordinate from the screen frame to the simulation frame
    TV coordinatetranslation(TV coord_screen, T s)
    {
        return(TV(coord_screen[0]-0.5*(0.5-s)*720,coord_screen[1]-0.5*(0.5-s)*720)/(720*s));
    }
    
    //calculate the acceleration for circular motion at the point indicated by location.
    //By Simplifying the deduction, we can get the magnitude of acceleration equals to the norm of the position vector,
    //and the direction of acceleration of acceleration equals to the inverse direction of the position vector.
    TV acceleratationforcircular(TV location)
    {
        TV acceleration = -location;
        return acceleration;
    }

    //return the distance between pos1 and pos2
    float distance(TV pos1, TV pos2)
    {
        float dist = sqrt(pow(pos1[0]-pos2[0],2)+pow(pos1[1]-pos2[1],2));
        return dist;
    }

    //enter TVStack positions 
    //return the normalized velocities direction stack considering the position cohesion
    TVStack Cohesionacc(TVStack pos)
    {
        TVStack center = TVStack::Zero(dim,n);  
        TVStack velocitiesdirection = TVStack::Zero(dim,n);
        for(int i = 0; i<n; i++)
        {
            int num = 0;
            for(int j = 0; j<n; j++)
            {
                if(j!=i)
                {
                    if(distance(pos.col(i),pos.col(j))<=threshold)
                    { 
                         center.col(i) = center.col(i) + positions.col(j);
                        num++;
                    }
                }
            }
            if(num){center.col(i) = center.col(i)/num;}
            TV velocitiesdirectiontemp = center.col(i) - positions.col(i);
            velocitiesdirection.col(i) = velocitiesdirectiontemp.normalized(); 
        }      
        return velocitiesdirection;
    }
    //enter TVStack postions, then velocities
    //return the velocity alignment part of individual point i
    TVStack Alignmentacc(TVStack pos, TVStack vel)
    {
        TVStack velocityalign = TVStack::Zero(dim,n);  
        for(int i = 0; i<n; i++)
        {
            int numm = 0;
            for(int j = 0; j<n; j++)
            {
                if(j!=i)
                {
                    if(distance(pos.col(i),pos.col(j))<=threshold)
                    { 
                        velocityalign.col(i) = velocityalign.col(i) + vel.col(j);
                        numm++;
                    }
                }
            }
            TV velocityaligntemp = (numm > 0? velocityalign.col(i)/numm:TV(0,0))-vel.col(i); 
            velocityalign.col(i) = velocityaligntemp.normalized();
        }
              
        return velocityalign;
    }

    TVStack Separationacc(TVStack pos, TVStack vel)
    {
        TVStack sepforce=TVStack::Zero(dim,n);
        TV sep;
        for (int i=0; i<n; i++)
        {   
            int numb = 0; 
            for(int j=0; j<n; j++)
            {
                if ((j!=i)&&(distance(pos.col(i),pos.col(j)) <= threshold_collision))
                {
                    sep = pos.col(i)-pos.col(j);
                    sepforce.col(i) =sepforce.col(i) + (sep.norm() > 0?sep.normalized() /sep.norm():TV(0,0) );
                    // assume the force is in inverse propotion to the distance, this is the outcome after deriving
                    // i.e. the maginitude of force acting on i is 1/distance(i,j), the direction is pos.col(i)-pos.col(j) 
                    numb++;
                }
            } 
            if(numb){sepforce.col(i) = sepforce.col(i)/numb;}       
        }
   
        return sepforce;
    }

    TVStack CollisionAvoidacc(TV pos)
    {
        TV colliavoidacc;
        TV dist;
        dist = pos-obstaclecenter_sim;
            if (distance(pos,obstaclecenter_sim) <= obstacleradius_sim)
            {
                //colliavoidacc = 20*dist.normalized();
                colliavoidacc = dist.norm() > 0?dist.normalized() /pow(dist.norm(),2):TV(0,0);
            }                                
            else{
                colliavoidacc = dist.normalized() /(dist.norm()-obstacleradius_sim);   
             }

        return colliavoidacc;
    }//still exist some bugs that the birds still have the possibility to crush into the obstacle when the velocity is too large


    void updateBehavior(MethodTypes type1, IntegrationMethodTypes type2)
    {
        if (!update)
            return;
        switch (type1)
        {
        case FREEFALL:
            for (int i = 0; i < n; i++)
            {
                positions.col(i) = positions.col(i) + h_step * velocities.col(i);
                velocities.col(i) = velocities.col(i) + TV(0, h_step * g);
            }
            break;
        case CIRCULAR:

            for (int i = 0; i < n; i++)
                {
                    
                    velocities(0, i) = -positions(1, i)+0.5;
                    velocities(1, i) = positions(0, i)-0.5;
                }
            positionstemp = positions;
            velocitiestemp = velocities;
            switch (type2)
            {
            case BasicIntergration:            
                for (int j = 0; j < n; j++)
                {
                    positionstemp.col(j) = positions.col(j) + h_step * velocities.col(j);
                    velocities.col(j) = velocities.col(j) + h_step * acceleratationforcircular(positions.col(j));
                    positions.col(j) = positionstemp.col(j);
                }
                break;
                
            case SymplecticEuler:               
                for (int j = 0; j < n; j++)
                {
                    positions.col(j) = positions.col(j) + h_step * velocities.col(j);
                    velocities.col(j) = velocities.col(j) + h_step * acceleratationforcircular(positions.col(j));
                    //velocities.col(j) = velocities.col(j) + h_step * (acceleratationforcircular(positions.col(j))+10*CollisionAvoidacc(positions.col(j)));
               
                }

                break;
                
            case Explicitmidpoint:
                for (int i = 0; i < n; i++)
                {
                    positionstemp.col(i) = positions.col(i) + h_step * velocities.col(i);
                    velocitiestemp.col(i) = velocities.col(i) + h_step * acceleratationforcircular(positions.col(i));
                    positions.col(i) = positions.col(i) + h_step * velocitiestemp.col(i);
                    velocities.col(i) = velocities.col(i) + h_step * acceleratationforcircular(positionstemp.col(i));
                }
                break;
            }
            break;
        
        case COHESION:
            
            for(int i = 0; i<n; i++)
            {          
                positions.col(i) = positions.col(i) + h_step * velocities.col(i);
                velocities.col(i) = velocities.col(i) + h_step * (Cohesionacc(positions).col(i)-0.2*velocities.col(i));
                
            }
            
            break;
        case ALIGNMENT:
          
           for(int i = 0; i<n; i++)
           {
               positions.col(i) = positions.col(i) + h_step*velocities.col(i);
               velocities.col(i) = velocities.col(i) + h_step*(Cohesionacc(positions).col(i)
                                                              +Alignmentacc(positions,velocities).col(i));
           }
           break;
        case SEPARATION:
           for(int i = 0; i<n; i++)
           {
               positions.col(i) = positions.col(i) + h_step*velocities.col(i);
               velocities.col(i) = velocities.col(i) + h_step*(0.2*Separationacc(positions,velocities).col(i)
                                                              +0.4*Cohesionacc(positions).col(i)
                                                              +0.4*Alignmentacc(positions,velocities).col(i));
           }
           break;
        case LEADER:
            positions.col(0) = coordinatetranslation(mouseposition,scale);
            //velocities.col(0) = TV::Zero(dim, n).unaryExpr([&](T dummy) { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX); }); 
           for(int i = 1; i<n; i++)
           {
               positions.col(i) = positions.col(i) + h_step*velocities.col(i);
               velocities.col(i) = velocities.col(i) + h_step*(0.5*Separationacc(positions,velocities).col(i)
                                                              +Cohesionacc(positions).col(i)
                                                              +Alignmentacc(positions,velocities).col(i)
                                                              +10*(positions.col(0)-positions.col(i)).normalized()
                                                              -velocities.col(i)
                                                              +0.6*CollisionAvoidacc(positions.col(i)));
           }
           break;
     
     
        }

    }
    void pause()
        {
        update = !update;
        }
        TVStack getPositions()
        {
        return positions;
        }
};
#endif

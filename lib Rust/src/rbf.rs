use rand::{random, Rng};
use std::borrow::Borrow;
use std::ops::Deref;
use nalgebra::distance;
use ndarray::prelude::*;

fn mean(data: &Vec<f64>) -> Option<f64> {
    let sum = data.iter().sum::<f64>() as f64;
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f64),
        _ => None,
    }

}

fn std_deviation(data: &Vec<f64>) -> Option<f64> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data.iter().map(|value| {
                let diff = data_mean - (*value as f64);

                diff * diff
            }).sum::<f64>() / count as f64;

            Some(variance.sqrt())
        },
        _ => None
    }
}


fn compute_vector_mean(computed_vector: &Vec<Vec<f64>>) -> Vec<f64>
{
    let mut result = computed_vector.iter().sum();
    result = result/computed_vector.len() as f64;
    result
}

fn power(a: f64, power: i32) -> f64
{
    let mut result = a;
    for i in 1..power
    {
        result *= a;
    }
    result
}


fn get_distance(x1: &Vec<f64>, x2: &Vec<f64>) -> f64
{
    let mut sum = 0.0;


    for i in 0..x1.len()
    {
        sum += power(x1[i] - x2[i], 2);
    }

    sum.sqrt()
}

fn kmeans(X: &Vec<Vec<f64>>, k: i32, max_iters: i32) -> (Vec<f64>, Vec<f64>)
{
    let mut cluster_list: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 0]; 0]; 0];
    let mut rng = rand::thread_rng();
    let mut converged = false;
    let mut current_iter = 0;
    let mut centroids:Vec<Vec<f64>> = Vec::new();
    let mut centroids2:Vec<f64> = Vec::new();

    for i in 0..k as usize
    {

        let mut random_number = rand::thread_rng().gen_range(0, X.len()) as usize;


        centroids.push(X[random_number].clone());
    }



    while !converged && current_iter < max_iters
    {
        cluster_list = vec![vec![vec![0.0; 0]; 0]; 0];


        for i in 0..(&centroids).len()
        {
            let mut tmp: Vec<Vec<f64>> = Vec::new();
            cluster_list.push(tmp);
        }



        for x in X
        {
            let mut lower_distance = f64::INFINITY;

            for c in &centroids
            {
                let mut current_distance = get_distance(c, x);

                if  current_distance < lower_distance {
                    lower_distance = current_distance;
                }

            }
            if lower_distance != f64::INFINITY
            {
                cluster_list[lower_distance as usize].push(x.clone());
            }
        }

        cluster_list.retain(|x| !x.is_empty() );

        let mut prev_centroids = centroids.clone();



        for j in 0..cluster_list.len()
        {
            centroids2.push(compute_vector_mean(&cluster_list[j]));
        }


        let mut sum_prev_centroids = 0.0;

        for i in prev_centroids
        {
            let tmp:f64  = i.iter().sum();
            sum_prev_centroids = sum_prev_centroids + tmp
        }
        let total_sum_centroids:f64 = centroids2.iter().sum();
        let pattern: f64 =  (sum_prev_centroids - total_sum_centroids).abs();

        converged = (pattern == 0.0);

        current_iter += 1
    }



    let mut result: Vec<f64> = vec![0.0; 0];

    for i in cluster_list
    {
        result.push(std_deviation(&i).unwrap());
    }


    (centroids2, result)

}


pub struct RBF {

    X:Vec<Vec<f64>>,
    y:Vec<Vec<f64>>,
    tX:Vec<Vec<f64>>,
    ty:Vec<Vec<f64>>,
    num_of_classes: i32,
    k: i32,
    std_from_clusters: bool

}


impl RBF {
    pub fn new(new_X:Vec<Vec<f64>>, new_y:Vec<Vec<f64>>, new_tX:Vec<Vec<f64>>, new_ty:Vec<Vec<f64>>, new_num_of_classes: i32, new_k: i32, new_std_from_clusters: bool) -> RBF{
        RBF {
            X:new_X,
            y:new_y,
            tX:new_tX,
            ty:new_ty,
            num_of_classes: new_num_of_classes,
            k: new_k,
            std_from_clusters: new_std_from_clusters
        }

    }

    pub fn init(new_X:Vec<Vec<f64>>, new_y:Vec<Vec<f64>>, new_tX:Vec<Vec<f64>>, new_ty:Vec<Vec<f64>>, new_num_of_classes: i32, new_k: i32) -> RBF{
        RBF {
            X:new_X,
            y:new_y,
            tX:new_tX,
            ty:new_ty,
            num_of_classes: new_num_of_classes,
            k: new_k,
            std_from_clusters: false
        }

    }

    pub fn convert_to_one_hot(x: &Vec<f64>, num_of_classes: i32) -> Vec<Vec<f64>>
    {
        let mut returned_result = vec![vec![0.0; num_of_classes as usize]; x.len()];
        for i in 0..x.len()
        {
            let c = x[i].clone();
            returned_result[i][c as usize] = 1.0
        }

        returned_result
    }

    pub fn rbf(x: &Vec<f64>, c: &Vec<f64>, s: f64) -> f64
    {
        let distance = get_distance(x, c);
        1.0 / (-distance/power(s, 2)).exp()
    }

    pub fn rbf_list(X: Vec<Vec<f64>>, centroids: Vec<f64>, std_list: Vec<f64>)
    {
        let mut RBF_list:Vec<Vec<f64>> = Vec::new();



        for x in X{
            RBF_list = centroids.iter().zip(std_list.iter()).map(|a| RBF::rbf(&x, a.0)).collect();
        }
    }
}
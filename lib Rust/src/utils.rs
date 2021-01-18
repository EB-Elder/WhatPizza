use rand::{random, Rng};
use std::borrow::Borrow;
use std::ops::Deref;
use nalgebra::distance;
use ndarray::prelude::*;
use std::slice::{from_raw_parts};

pub fn power(a: f64, power: i32) -> f64
{
    let mut result = a;
    for i in 1..power
    {
        result *= a;
    }
    result
}

pub fn sign(value: f64) -> f64
{
    if value >= 0.0
    {
        return 1.0;
    }
    return -1.0;

}


pub fn weight_array_1dto3d(model:  &Vec<f64>, npl:  &[i32]) -> Vec<Vec<Vec<f64>>>
{
    let mut result = vec![vec![vec![0.0; 0]]];

    let mut counter:usize = 0;
    for l in 0..npl.len()
    {

        if l == 0
        {
            result[0] = vec![vec![model[counter];1]];
            counter+=1;
            continue
        }
        let mut tmp0:Vec<Vec<f64>> = Vec::new();
        for i in 0..npl[l-1]+1
        {
            let mut tmp1:Vec<f64>= Vec::new();
            for j in 0..npl[l] + 1
            {
                tmp1.push(model[counter]);
                counter+=1;
            }
            tmp0.push(tmp1);
        }
        result.push(tmp0);
    }

    result
}

pub fn weight_array_3dto1d(model:  &mut Vec<f64>,vec_boxed_model:  &Vec<Vec<Vec<f64>>>, npl:  &[i32])
{
    let mut counter:usize = 0;
    let mut boxed_model;

    unsafe {
        boxed_model = &mut *model;
    }

    for l in 0..npl.len()
    {

        if l == 0
        {
            counter+=1;
            continue
        }
        for i in 0..npl[l-1]+1
        {
            for j in 0..npl[l] + 1
            {
                boxed_model[counter] = vec_boxed_model[l][i as usize][j as usize];
                counter+=1;
            }
        }
    }
}

pub fn _predict_linear_model(model: *mut Vec<f64>, inputs: &Vec<f64>, inputs_size: usize,  is_classification: bool) -> f64
{
    let boxed_model;

    unsafe {
        //Récupération des contenus des pointeurs
        boxed_model = model.as_ref().unwrap();
    }

    let mut sum = 0.0;

    //prédictions
    for i  in 0..inputs.len() {
        sum += inputs[i] * boxed_model[i];
    }

    if is_classification
    {
        return sign(sum);
    }
    sum


}

pub fn _1dto2dVec(v: &Vec<f64>, row: usize, col: usize) -> Vec<Vec<f64>>
{
    let mut result:Vec<Vec<f64>> = Vec::new();
    let mut count:usize = 0;
    for i in 0..col
    {
        let mut tmp = Vec::new();
        for j in 0..row
        {
            tmp.push(v[count]);
            count += 1
        }
        result.push(tmp);
    }

    result
}

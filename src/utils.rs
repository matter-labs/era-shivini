use boojum::cs::implementations::utils::domain_generator_for_size;

use super::*;

#[inline(always)]
pub fn bitreverse_index(n: usize, l: usize) -> usize {
    let mut r = n.reverse_bits();
    // now we need to only use the bits that originally were "last" l, so shift

    r >>= (std::mem::size_of::<usize>() * 8) - l;

    r
}

pub fn bitreverse(input: &mut [F]) {
    use boojum::fft::bitreverse_enumeration_inplace;
    bitreverse_enumeration_inplace(input);
}

#[allow(dead_code)]
pub fn divide_by_vanishing_poly_in_bitreversed(
    poly: &mut Poly<LDE>,
    domain_size: usize,
    lde_degree: usize,
) -> CudaResult<()> {
    // p(x)/ z(x) = p(x) / x^n -1
    let mut inv_coset_shifts_in_domain = compute_coset_powers_bitreversed(domain_size, lde_degree);
    for el in inv_coset_shifts_in_domain.iter_mut() {
        let mut tmp = el.pow_u64(domain_size as u64);
        tmp.sub_assign(&F::ONE);
        *el = tmp.inverse().unwrap();
    }

    for (coset_values, denum) in poly
        .storage
        .as_mut()
        .chunks_mut(domain_size)
        .zip(inv_coset_shifts_in_domain.into_iter())
    {
        let d_denum: DF = denum.into();
        arith::scale(coset_values, &d_denum)?;
    }

    Ok(())
}

pub fn divide_by_vanishing_poly_over_coset<'a>(
    poly: &mut Poly<'a, CosetEvaluations>,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
) -> CudaResult<()> {
    // p(x)/ z(x) = p(x) / x^n -1
    let mut inv_coset_shifts_in_domain = compute_coset_powers_bitreversed(domain_size, lde_degree);
    for el in inv_coset_shifts_in_domain.iter_mut() {
        let mut tmp = el.pow_u64(domain_size as u64);
        tmp.sub_assign(&F::ONE);
        *el = tmp.inverse().unwrap();
    }

    let scalar = inv_coset_shifts_in_domain[coset_idx].into();
    poly.scale(&scalar)?;

    Ok(())
}

#[allow(dead_code)]
pub fn compute_l0_poly_lde_in_bitreversed(
    domain_size: usize,
    lde_degree: usize,
    quotient_degree: usize,
) -> CudaResult<Poly<'static, LDE>> {
    // z(x) / (x-1) = x^n -1 / (x-1)
    let mut domain_elems = dvec!(domain_size);
    helpers::compute_domain_elems(&mut domain_elems, domain_size)?;
    ntt::bitreverse(&mut domain_elems)?;

    let coset_shifts =
        compute_coset_powers_bitreversed(domain_size, lde_degree)[..quotient_degree].to_vec();

    let mut l0_values = dvec!(quotient_degree * domain_size);

    let one = DF::one()?;
    for (shift, coset_values) in coset_shifts
        .into_iter()
        .zip(l0_values.chunks_mut(domain_size))
    {
        let d_shift: DF = shift.into();
        mem::d2d(&domain_elems, coset_values)?;
        arith::scale(coset_values, &d_shift)?;
        arith::sub_constant(coset_values, &one)?;
        arith::inverse(coset_values)?;
        let mut shift_in_domain = shift.pow_u64(domain_size as u64);
        shift_in_domain.sub_assign(&F::ONE);
        let d_shift_in_n = shift_in_domain.into();
        arith::scale(coset_values, &d_shift_in_n)?;
    }

    Ok(Poly::from(l0_values))
}

pub fn compute_l0_over_coset(
    coset_idx: usize,
    domain_size: usize,
    quotient_degree: usize,
) -> CudaResult<Poly<'static, CosetEvaluations>> {
    // z(x) / (x-1) = x^n -1 / (x-1)
    let mut domain_elems = dvec!(domain_size);
    helpers::compute_domain_elems(&mut domain_elems, domain_size)?;
    ntt::bitreverse(&mut domain_elems)?;

    let coset_shifts = compute_coset_powers_bitreversed(domain_size, quotient_degree);

    let mut coset_values = dvec!(domain_size);
    let shift = coset_shifts[coset_idx];
    let d_shift: DF = shift.into();
    let one = DF::one()?;
    mem::d2d(&domain_elems, &mut coset_values)?;
    arith::scale(&mut coset_values, &d_shift)?;
    arith::sub_constant(&mut coset_values, &one)?;
    arith::inverse(&mut coset_values)?;
    let mut shift_in_domain = shift.pow_u64(domain_size as u64);
    shift_in_domain.sub_assign(&F::ONE);
    let d_shift_in_n = shift_in_domain.into();
    arith::scale(&mut coset_values, &d_shift_in_n)?;

    Ok(Poly::from(coset_values))
}

pub fn compute_coset_powers_bitreversed(domain_size: usize, lde_degree: usize) -> Vec<F> {
    let lde_gen = domain_generator_for_size((lde_degree * domain_size) as u64);
    let mut coset_shifts = vec![F::multiplicative_generator(); lde_degree];
    let mut acc = lde_gen;
    for base in coset_shifts[1..].iter_mut() {
        base.mul_assign(&acc);
        acc.mul_assign(&lde_gen);
    }
    crate::utils::bitreverse(&mut coset_shifts);

    coset_shifts
}

#[allow(dead_code)]
pub fn compute_omega_values_lde(
    domain_size: usize,
    lde_degree: usize,
) -> CudaResult<Poly<'static, LDE>> {
    use boojum::cs::implementations::utils::*;
    let mut omega_values = dvec!(domain_size);
    helpers::compute_domain_elems(&mut omega_values, domain_size)?;
    ntt::bitreverse(&mut omega_values)?;

    let lde_generator = domain_generator_for_size::<F>((domain_size * lde_degree) as u64);
    let mut lde_generators = materialize_powers_serial::<F, Global>(lde_generator, lde_degree);
    crate::utils::bitreverse(&mut lde_generators);
    let shift = F::multiplicative_generator();
    let mut d_lde_generators = vec![];
    for gen in lde_generators.iter_mut() {
        gen.mul_assign(&shift);
        d_lde_generators.push(gen.clone().into());
    }

    let mut omega_values_lde = dvec!(domain_size * lde_degree);
    for (lde_gen, coset) in d_lde_generators
        .into_iter()
        .zip(omega_values_lde.chunks_exact_mut(domain_size))
    {
        mem::d2d(&omega_values, coset)?;
        arith::scale(coset, &lde_gen)?;
    }

    Ok(Poly::from(omega_values_lde))
}

pub fn compute_omega_values_for_coset(
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
) -> CudaResult<Poly<'static, CosetEvaluations>> {
    use boojum::cs::implementations::utils::*;
    let mut omega_values = dvec!(domain_size);
    helpers::compute_domain_elems(&mut omega_values, domain_size)?;
    ntt::bitreverse(&mut omega_values)?;

    let lde_generator = domain_generator_for_size::<F>((domain_size * lde_degree) as u64);
    let mut lde_generators = materialize_powers_serial::<F, Global>(lde_generator, lde_degree);
    crate::utils::bitreverse(&mut lde_generators);
    let shift = F::multiplicative_generator();
    let mut d_lde_generators = vec![];
    for gen in lde_generators.iter_mut() {
        gen.mul_assign(&shift);
        d_lde_generators.push(gen.clone().into());
    }

    let lde_gen_for_coset = &d_lde_generators[coset_idx];
    arith::scale(&mut omega_values, lde_gen_for_coset)?;

    Ok(Poly::from(omega_values))
}
pub fn assert_adjacent<T>(src: &[&[T]]) {
    if src.is_empty() {
        return;
    }
    let mut prev_ptr = src[0].as_ref().as_ptr();
    let domain_size = src[0].len();
    unsafe {
        for p in src[1..].iter() {
            assert_eq!(p.len(), domain_size);
            let this_ptr = prev_ptr.add(domain_size);
            assert!(std::ptr::eq(p.as_ref().as_ptr(), this_ptr));
            prev_ptr = this_ptr;
        }
    }
}

pub fn assert_adjacent_base<P: PolyForm>(src: &[Poly<P>]) {
    let inner_values: Vec<_> = src.iter().map(|p| p.storage.as_ref()).collect();
    assert_adjacent(&inner_values);
}

pub fn assert_multiset_adjacent_base<P: PolyForm>(src: &[&[Poly<P>]]) {
    let mut flattened = vec![];
    for p in src.iter() {
        for sp in p.iter() {
            flattened.push(sp.storage.as_ref());
        }
    }
    assert_adjacent(&flattened[..])
}

pub fn assert_adjacent_ext<P: PolyForm>(src: &[ComplexPoly<P>]) {
    let mut flattened = vec![];
    for p in src.iter() {
        flattened.push(p.c0.storage.as_ref());
        flattened.push(p.c1.storage.as_ref());
    }
    assert_adjacent(&flattened[..])
}

pub fn assert_multiset_adjacent_ext<P: PolyForm>(src: &[&[ComplexPoly<P>]]) {
    let mut flattened = vec![];
    for p in src.iter() {
        for sp in p.iter() {
            flattened.push(sp.c0.storage.as_ref());
            flattened.push(sp.c1.storage.as_ref());
        }
    }
    assert_adjacent(&flattened[..])
}

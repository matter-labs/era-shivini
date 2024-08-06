use super::*;

use boojum::cs::implementations::pow::PoWRunner;
use era_cudart::slice::DeviceVariable;

pub struct DeviceBlake2sPOW;

impl PoWRunner for DeviceBlake2sPOW {
    fn run_from_bytes(h_seed: Vec<u8>, pow_bits: u32, _worker: &boojum::worker::Worker) -> u64 {
        use era_cudart::slice::DeviceSlice;
        let _seed_len = h_seed.len();
        let unit_len = std::mem::size_of::<F>();
        assert_eq!(h_seed.len() % unit_len, 0);
        let num_elems = h_seed.len() / unit_len;
        let mut seed = svec!(num_elems);
        seed.copy_from_slice(&h_seed).unwrap();

        let challenge = unsafe {
            let seed = DeviceSlice::from_slice(&seed);
            let mut result = DF::zero().unwrap();
            let result_as_mut_slice =
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut _, 1);
            let result_as_var = DeviceVariable::from_mut_slice(result_as_mut_slice);
            if !is_dry_run().unwrap() {
                boojum_cuda::blake2s::blake2s_pow(
                    seed,
                    pow_bits,
                    u64::MAX,
                    result_as_var,
                    get_stream(),
                )
                .expect("pow on device");
            }
            let h_result: F = result.into();
            h_result.0
        };

        if !is_dry_run().unwrap() {
            assert!(Self::verify_from_bytes(h_seed, pow_bits, challenge));
        }

        challenge
    }

    fn verify_from_bytes(seed: Vec<u8>, pow_bits: u32, challenge: u64) -> bool {
        use blake2::Blake2s256;
        use blake2::Digest;

        let mut new_transcript = Blake2s256::new();
        new_transcript.update(&seed);
        new_transcript.update(&challenge.to_le_bytes());
        let mut le_bytes = [0u8; 8];
        le_bytes.copy_from_slice(&new_transcript.finalize().as_slice()[..8]);

        u64::from_le_bytes(le_bytes).trailing_zeros() >= pow_bits
    }
}

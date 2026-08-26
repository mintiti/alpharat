fn main() {
    alpharat_mcgs::gc::init();
    pyrat_sdk::run(
        alpharat_mcgs_bot::McgsBot::new(),
        "alpharat-mcgs",
        "alpharat",
    );
    alpharat_mcgs::gc::shutdown();
}

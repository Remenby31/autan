# Changelog

release-please writes this file from the squashed pull request titles on `main`. Edit an entry in the release pull request, before it merges. Versions follow [SemVer](https://semver.org/spec/v2.0.0.html): below 1.0.0, a breaking change bumps the minor version, and a `feat` or `fix` the patch.

## 0.1.0 (2026-10-31)


### Added

* **core:** face-based polyhedral mesh, FP64 geometry and mesh contract ([#11](https://github.com/Remenby31/autan/issues/11)) ([7334f3f](https://github.com/Remenby31/autan/commit/7334f3fd340d88451b00c9aafbdd50c304b9bd2c))
* **core:** mesh quality report, RCM renumbering and autan-mesh-info ([#34](https://github.com/Remenby31/autan/issues/34)) ([f501a40](https://github.com/Remenby31/autan/commit/f501a40c03213c444ffc9d18549f9d798648dc2e))
* **io:** read Gmsh 4.1, Fluent and OpenFOAM polyMesh meshes ([#32](https://github.com/Remenby31/autan/issues/32)) ([e23268d](https://github.com/Remenby31/autan/commit/e23268dff0fcf2e6cab510113a136bea8029b33c))
* **partition:** METIS/Scotch partitioning and GPU-aware MPI halo exchange ([#40](https://github.com/Remenby31/autan/issues/40)) ([2dc623e](https://github.com/Remenby31/autan/commit/2dc623e877968aa83528ce139d11286e99e6d2eb))
* **sdk:** add autan ui and the ui extra ([#52](https://github.com/Remenby31/autan/issues/52)) ([f83d0f2](https://github.com/Remenby31/autan/commit/f83d0f23882c8c9c90ceed1a34ed5c443173e848))
* **sdk:** fake backend, autan.Results API and the Autan rename ([#24](https://github.com/Remenby31/autan/issues/24)) ([47471e1](https://github.com/Remenby31/autan/commit/47471e112ee05e914833a140c21c1a0813a8481f))
* **sdk:** read solution.cgns on first use and export autan.RunInfo ([#37](https://github.com/Remenby31/autan/issues/37)) ([0da71ab](https://github.com/Remenby31/autan/commit/0da71ab27c970851c5425ce0cee00e26e080c8fc))
* **sdk:** register the mesh core diagnostic codes E0101-E0135 ([#35](https://github.com/Remenby31/autan/issues/35)) ([4776358](https://github.com/Remenby31/autan/commit/4776358ea5dc080c3fa59001f9cb9e930555a50f))
* **sdk:** typed case schema, diagnostics registry and cfd plan/doctor/version CLI ([#12](https://github.com/Remenby31/autan/issues/12)) ([751aedf](https://github.com/Remenby31/autan/commit/751aedfb760d454b2adae4ac7218ad85315aea93))
* **ui:** web UI skeleton, local server and WebGPU surface viewer ([#36](https://github.com/Remenby31/autan/issues/36)) ([ded5151](https://github.com/Remenby31/autan/commit/ded515165b19b87b3578e99dcad63e7bee595f58))
* **vv:** manufactured solutions, observed-order tool, mesh families and reference FV solve ([#9](https://github.com/Remenby31/autan/issues/9)) ([3068228](https://github.com/Remenby31/autan/commit/3068228e02f7231dcfffeb1e6de7fa6b5af2e1e9))


### Fixed

* **ci:** accept DrivAer, three.js and CMake names in typos ([#33](https://github.com/Remenby31/autan/issues/33)) ([f4f77ed](https://github.com/Remenby31/autan/commit/f4f77edced1684e35a5eaaeaa0ac5b5fef6ad789))
* **ci:** accept pydantic's ser identifier in typos ([#22](https://github.com/Remenby31/autan/issues/22)) ([454e304](https://github.com/Remenby31/autan/commit/454e304b2ec20aeafa8811d803790294bd0c2db5))
* **ci:** exclude French translations from typos ([#58](https://github.com/Remenby31/autan/issues/58)) ([edcface](https://github.com/Remenby31/autan/commit/edcface90cf6d0cbce1c8915a6104bb32f9af759))
* **ci:** keep test data byte for byte ([#26](https://github.com/Remenby31/autan/issues/26)) ([9261daa](https://github.com/Remenby31/autan/commit/9261daa8975ef7d7457c2008059053a71234c8cd))
* **ci:** run ruff on Python files only and ignore typos crate tags ([#16](https://github.com/Remenby31/autan/issues/16)) ([c9a3ca2](https://github.com/Remenby31/autan/commit/c9a3ca228057b675fd6ebb8d203039bf99ccf2e2))


### Documentation

* adopt engineering rules for agents and contributors ([d769576](https://github.com/Remenby31/autan/commit/d769576055d045768df80851ba1c9a0fe2f3222a))
* adopt repository standards, community files, lint and CI ([#6](https://github.com/Remenby31/autan/issues/6)) ([6aefc03](https://github.com/Remenby31/autan/commit/6aefc03f3fbd5d87859f05a0187900a6c3fb2735))
* **agents:** allow NVIDIA CUDA runtime wheels as dependencies ([#42](https://github.com/Remenby31/autan/issues/42)) ([0199235](https://github.com/Remenby31/autan/commit/0199235439a86fc65f0f6e234bcc146112266f59))
* **contracts:** freeze case schema, MeshInfo and results contracts ([#7](https://github.com/Remenby31/autan/issues/7)) ([6199ecc](https://github.com/Remenby31/autan/commit/6199ecc077f8a804026be933e1d99bcc17219220))
* **design:** align run states with the results contract ([#39](https://github.com/Remenby31/autan/issues/39)) ([95b04e8](https://github.com/Remenby31/autan/commit/95b04e83cfadb7ae3b3ce0d5a18e81fbcece419c))
* **design:** project logo ([#27](https://github.com/Remenby31/autan/issues/27)) ([03b99fb](https://github.com/Remenby31/autan/commit/03b99fb76d3915c110d9e0a30a93074a57089a43))
* **design:** UI inspiration and visual direction ([#19](https://github.com/Remenby31/autan/issues/19)) ([54a210b](https://github.com/Remenby31/autan/commit/54a210b0e8e5a38f60dd3a0dc6e9c5c99a3a56d7))
* kernel generator design research ([#10](https://github.com/Remenby31/autan/issues/10)) ([ccfca09](https://github.com/Remenby31/autan/commit/ccfca090cd8c8f53bc2ab6c11c3bef3075e49ecc))
* name the project Autan ([#25](https://github.com/Remenby31/autan/issues/25)) ([e2e2e0f](https://github.com/Remenby31/autan/commit/e2e2e0fa2ee800f468929ac0adb2d09f7c3ff6b5))
* **partition:** order cells by their global-mesh index and keep their global ids ([#54](https://github.com/Remenby31/autan/issues/54)) ([adf15ab](https://github.com/Remenby31/autan/commit/adf15abc7aa3b982415593dcebe1263daee006e9))
* **plan:** add Metal backend milestone and ARM-optimised CPU backend ([#20](https://github.com/Remenby31/autan/issues/20)) ([0502266](https://github.com/Remenby31/autan/commit/050226674685b0d9a4e1f1b79e04cc6d99e0371a))
* **plan:** fix the source links ([#60](https://github.com/Remenby31/autan/issues/60)) ([4221433](https://github.com/Remenby31/autan/commit/422143310b4949f0a4feaa6284e87bd67a40a372))
* **plan:** make in-house AD primary, Enzyme an oracle ([#13](https://github.com/Remenby31/autan/issues/13)) ([db21096](https://github.com/Remenby31/autan/commit/db210963a8fc336accce6b2ec6f2cc6c990f6a54))
* **plan:** use the Autan name in the source tree ([#30](https://github.com/Remenby31/autan/issues/30)) ([69f8a38](https://github.com/Remenby31/autan/commit/69f8a3808bd5648fb92fb113731f3f033b5124f4))
* **readme:** link the documentation site and update the status ([#47](https://github.com/Remenby31/autan/issues/47)) ([01cf8e0](https://github.com/Remenby31/autan/commit/01cf8e0bc04a18534e7a72441d08680524d31c98))
* **readme:** present Autan with current facts, stack badges and its origin ([#18](https://github.com/Remenby31/autan/issues/18)) ([01e376a](https://github.com/Remenby31/autan/commit/01e376a493f57ca035e1eda436d8df42188eb9f1))
* **site:** add the Fumadocs documentation site and its Pages deployment ([#31](https://github.com/Remenby31/autan/issues/31)) ([930db2c](https://github.com/Remenby31/autan/commit/930db2c1374bae83f95648dcd074adc37451d71e))
* **site:** show the Autan symbol next to the wordmark ([#46](https://github.com/Remenby31/autan/issues/46)) ([b2f530f](https://github.com/Remenby31/autan/commit/b2f530f24d48a2bab268d920a69cc11b31dc849d))
* use the Autan name in page titles ([#45](https://github.com/Remenby31/autan/issues/45)) ([a2b4e65](https://github.com/Remenby31/autan/commit/a2b4e6599d8b1e2a8c8b08559c7789138895138b))
* validation dataset catalogue ([#23](https://github.com/Remenby31/autan/issues/23)) ([a2c01d5](https://github.com/Remenby31/autan/commit/a2c01d54c6cc0d80f6ff360345b84b8563667011))

## 0.0.0

Starting point of initial development. Nothing is released.

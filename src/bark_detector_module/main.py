import asyncio

from viam.components.sensor import Sensor
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .detector import Detector


def _register() -> None:
    Registry.register_resource_creator(
        Sensor.API,
        Detector.MODEL,
        ResourceCreatorRegistration(Detector.new, Detector.validate_config),
    )


async def main() -> None:
    _register()
    module = Module.from_args()
    module.add_model_from_registry(Sensor.API, Detector.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

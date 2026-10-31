"""Helper for Home Assistant Idasen Desk integration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging

from bleak.backends.device import BLEDevice
from bleak.exc import BleakCharacteristicNotFoundError, BleakError
from idasen import _UUID_REFERENCE_INPUT, IdasenDesk, _meters_to_bytes

from .connection_manager import ConnectionManager

_LOGGER = logging.getLogger(__name__)

_MOVE_TIMEOUT_S = 35  # max travel ~65cm at ~2cm/s = ~32.5s
_MOVE_TOLERANCE_M = 0.001  # 1 mm — maximum byte accuracy
_MOVE_LOOP_INTERVAL_S = 0.4


class Desk:
    """Wrapper around the IdasenDesk from the idasen library."""

    def __init__(
        self,
        update_callback: Callable[[int | None], None] | None,
        monitor_height: bool = True,
    ) -> None:
        """Initialize the wrapper.

        Args:
            update_callback: Called with the current height percent whenever
                height changes or connection state changes.
            monitor_height: Whether to subscribe to BLE height notifications
                after connecting. When True (default), height is tracked
                continuously and precise position-based movement is used.
                When False, height is never read and move_to falls back to
                idasen's polling-based move_to_target — a degraded mode that
                generates significantly more GATT traffic and is unreliable
                over ESPHome BLE proxies.

        """
        self._ble_device: BLEDevice | None = None
        self._connection_manager: ConnectionManager | None = None
        self._height: float | None = None
        self._monitor_height: bool = monitor_height
        self._move_task: asyncio.Task[None] | None = None

        if update_callback:
            self._update_callback = update_callback
        else:

            def empty_update_callback(height: int | None) -> None:
                pass

            self._update_callback = empty_update_callback

    async def connect(self, ble_device: BLEDevice, retry: bool = True) -> None:
        """Perform the bluetooth connection to the desk."""
        _LOGGER.debug("Connecting")

        if (
            self._connection_manager is None
            or self._ble_device is None
            or ble_device.address != self._ble_device.address
        ):
            _LOGGER.debug("Initializing idasen desk")
            await self.disconnect()
            self._ble_device = ble_device
            self._connection_manager = self._create_connection_manager(self._ble_device)

        await self._connection_manager.connect(retry)

    async def disconnect(self) -> None:
        """Disconnect from the desk."""
        if self._connection_manager:
            _LOGGER.debug("Disconnecting")
            await self._connection_manager.disconnect()

    async def move_to(self, heigh_percent: int) -> None:
        """Move the desk to a specific position."""
        _LOGGER.debug("Moving to %s", heigh_percent)
        if not self.is_connected or self._idasen_desk is None:
            _LOGGER.warning("Not connected")
            return

        # Only one move may run at a time; the newest request wins.
        await self._cancel_move()
        task = asyncio.create_task(self._move_to(self._idasen_desk, heigh_percent))
        self._move_task = task
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            _LOGGER.debug("Move to %s%% interrupted", heigh_percent)
        finally:
            if self._move_task is task:
                self._move_task = None

    async def _cancel_move(self) -> None:
        """Cancel the in-progress move, if any, and wait for it to end."""
        while (task := self._move_task) is not None and not task.done():
            task.cancel()
            await asyncio.wait([task])

    async def _move_to(self, desk: IdasenDesk, heigh_percent: int) -> None:
        if desk.is_moving:
            try:
                await desk.stop()
            except BleakError, TimeoutError:
                _LOGGER.warning("Failed to stop desk before moving", exc_info=True)
                return
            # Let it settle before requesting a new move
            await asyncio.sleep(0.5)

        height = IdasenDesk.MIN_HEIGHT + (
            IdasenDesk.MAX_HEIGHT - IdasenDesk.MIN_HEIGHT
        ) * (heigh_percent / 100)

        if self._height is not None and abs(self._height - height) < _MOVE_TOLERANCE_M:
            return

        if not self._monitor_height:
            try:
                await desk.wakeup()
                await desk.move_to_target(height)
            except BleakError, TimeoutError:
                _LOGGER.warning("Failed to move desk", exc_info=True)
            return

        if self._height is None:
            _LOGGER.warning(
                "Cannot move: current height unknown, no notification received yet"
            )
            return

        # Wake the desk and stop any residual movement before starting
        try:
            await desk.wakeup()
            await desk.stop()
        except BleakError, TimeoutError:
            _LOGGER.warning("Failed to prepare desk for moving", exc_info=True)
            return

        # Write the target position to the desk's position-control characteristic.
        # The desk firmware handles closed-loop deceleration and stopping — more
        # accurate than directional commands, and no speed reads needed during the
        # move. Height comes from the monitor notifications.
        reference_data = _meters_to_bytes(height)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _MOVE_TIMEOUT_S
        try:
            while loop.time() < deadline:
                try:
                    await desk._client.write_gatt_char(  # noqa: SLF001
                        _UUID_REFERENCE_INPUT, reference_data, response=False
                    )
                except BleakCharacteristicNotFoundError:
                    # Service map is already populated at connect time, so a missing
                    # characteristic mid-move means the connection is degrading.
                    # Abort and let the disconnect callback handle reconnection.
                    _LOGGER.warning(
                        "Position-control characteristic not found mid-move; "
                        "connection is likely dropping. Move aborted."
                    )
                    return
                await asyncio.sleep(_MOVE_LOOP_INTERVAL_S)
                current = self._height
                if current is not None and abs(current - height) < _MOVE_TOLERANCE_M:
                    break
            else:
                _LOGGER.warning("Move timed out reaching %s%%", heigh_percent)
        except BleakError, TimeoutError:
            _LOGGER.warning("Failed to move desk", exc_info=True)

    async def move_up(self) -> None:
        """Move the desk up."""
        _LOGGER.debug("Moving up")
        await self.move_to(100)

    async def move_down(self) -> None:
        """Move the desk down."""
        _LOGGER.debug("Moving down")
        await self.move_to(0)

    async def stop(self) -> None:
        """Stop moving the desk."""
        _LOGGER.debug("Stopping")
        if not self.is_connected or self._idasen_desk is None:
            _LOGGER.warning("Not connected")
            return
        # End our move loop first, or it would resend the target after the stop.
        await self._cancel_move()
        await self._idasen_desk.stop()

    async def _start_monitoring(self) -> None:
        """Start monitoring for height changes."""
        if not self.is_connected or self._idasen_desk is None:
            _LOGGER.warning("Not connected")
            return

        async def update_height(height: float) -> None:
            self._height = height
            self._update_callback(self.height_percent)

        await self._idasen_desk.monitor(update_height)

    @property
    def height(self) -> float | None:
        """The current height in meters."""
        return self._height

    @property
    def height_percent(self) -> int | None:
        """The current height in percentage."""
        if self._height is None:
            return None

        return int(
            round(
                100
                * (self._height - IdasenDesk.MIN_HEIGHT)
                / (IdasenDesk.MAX_HEIGHT - IdasenDesk.MIN_HEIGHT)
            )
        )

    @property
    def is_connected(self) -> bool:
        """True if the bluetooth connection is currently established."""
        if self._idasen_desk is None:
            return False
        # bleak `is_connected` method returns a `_DeprecatedIsConnectedReturn`,
        # so we properly cast it to bool otherwise `is_connected == True`
        # will always be False.
        return bool(self._idasen_desk.is_connected)

    @property
    def _idasen_desk(self) -> IdasenDesk | None:
        if self._connection_manager is None:
            return None
        return self._connection_manager.idasen_desk

    def _create_connection_manager(self, ble_device: BLEDevice) -> ConnectionManager:
        async def connect_callback() -> None:
            _LOGGER.debug("Connect callback called")
            if self._idasen_desk is None:
                _LOGGER.error("Desk is None after connecting")
                return

            if self._monitor_height:
                self._height = await self._idasen_desk.get_height()
                await self._start_monitoring()

            self._update_callback(self.height_percent)

        return ConnectionManager(
            ble_device,
            connect_callback=connect_callback,
            disconnect_callback=lambda: self._update_callback(self.height_percent),
        )

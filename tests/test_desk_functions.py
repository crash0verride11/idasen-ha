"""Tests for idasen_ha."""

import asyncio
from unittest import mock
from unittest.mock import MagicMock, Mock

from bleak.exc import BleakCharacteristicNotFoundError, BleakError
from idasen import _UUID_REFERENCE_INPUT, _meters_to_bytes
import pytest

from idasen_ha import Desk

from . import FAKE_BLE_DEVICE, height_percent_to_meters


@pytest.fixture(autouse=True)
def fast_move_loop(monkeypatch: pytest.MonkeyPatch):
    """Shorten the position-control loop interval so tests run quickly."""
    monkeypatch.setattr("idasen_ha._MOVE_LOOP_INTERVAL_S", 0.01)


async def connect_at(mock_idasen_desk: MagicMock, height_percent: int | None) -> Desk:
    """Connect a height-monitoring desk reporting the given starting height."""
    desk = Desk(None, True)
    mock_idasen_desk.get_height.return_value = (
        None if height_percent is None else height_percent_to_meters(height_percent)
    )
    await desk.connect(FAKE_BLE_DEVICE)
    mock_idasen_desk.is_moving = False
    return desk


def reach_target_on_write(desk: Desk, target_percent: int):
    """Return a write_gatt_char side effect that moves the desk to the target."""

    async def reach_target(char, data, **kwargs):
        if char == _UUID_REFERENCE_INPUT:
            desk._height = height_percent_to_meters(target_percent)

    return reach_target


async def test_monitor_height(mock_idasen_desk: MagicMock):
    """Test height monitoring."""
    update_callback = Mock()
    desk = Desk(update_callback, True)

    HEIGHT_PCT_1 = 50
    HEIGHT_MTS_1 = height_percent_to_meters(HEIGHT_PCT_1)
    mock_idasen_desk.get_height.return_value = HEIGHT_MTS_1

    await desk.connect(FAKE_BLE_DEVICE)
    mock_idasen_desk.establish_connection.assert_called()
    mock_idasen_desk.pair.assert_called()
    mock_idasen_desk.get_height.assert_called()
    update_callback.assert_called_with(HEIGHT_PCT_1)
    assert desk.height == HEIGHT_MTS_1
    assert desk.height_percent == HEIGHT_PCT_1

    HEIGHT_PCT_2 = 80
    HEIGHT_MTS_2 = height_percent_to_meters(HEIGHT_PCT_2)
    await mock_idasen_desk.trigger_monitor_callback(HEIGHT_MTS_2)
    update_callback.assert_called_with(HEIGHT_PCT_2)
    assert desk.height == HEIGHT_MTS_2
    assert desk.height_percent == HEIGHT_PCT_2


async def test_moves(mock_idasen_desk: MagicMock):
    """Test movement writes the target to the position-control characteristic."""
    desk = await connect_at(mock_idasen_desk, 0)
    client = mock_idasen_desk._client

    client.write_gatt_char.side_effect = reach_target_on_write(desk, 50)
    await desk.move_to(50)
    mock_idasen_desk.wakeup.assert_awaited()
    client.write_gatt_char.assert_awaited_with(
        _UUID_REFERENCE_INPUT,
        _meters_to_bytes(height_percent_to_meters(50)),
        response=False,
    )

    client.write_gatt_char.side_effect = reach_target_on_write(desk, 100)
    await desk.move_up()
    client.write_gatt_char.assert_awaited_with(
        _UUID_REFERENCE_INPUT,
        _meters_to_bytes(height_percent_to_meters(100)),
        response=False,
    )

    client.write_gatt_char.side_effect = reach_target_on_write(desk, 0)
    await desk.move_down()
    client.write_gatt_char.assert_awaited_with(
        _UUID_REFERENCE_INPUT,
        _meters_to_bytes(height_percent_to_meters(0)),
        response=False,
    )
    mock_idasen_desk.move_to_target.assert_not_called()


async def test_stop_before_move(mock_idasen_desk: MagicMock):
    """Test that movement is stopped before new movement starts."""
    desk = await connect_at(mock_idasen_desk, 0)
    mock_idasen_desk.is_moving = True
    mock_idasen_desk._client.write_gatt_char.side_effect = reach_target_on_write(
        desk, 50
    )

    await desk.move_to(50)
    # stop() before move, then wakeup + stop (prepare), then reference input
    mock_idasen_desk.assert_has_calls(
        [mock.call.stop(), mock.call.wakeup(), mock.call.stop()]
    )
    mock_idasen_desk._client.write_gatt_char.assert_awaited()
    # no trailing stop() — desk firmware handles stopping at target
    assert mock_idasen_desk.stop.call_count == 2


async def test_stop_failure_before_move(mock_idasen_desk: MagicMock):
    """Test that move_to returns early when stop raises before moving."""
    desk = Desk(None, False)
    await desk.connect(FAKE_BLE_DEVICE)

    mock_idasen_desk.is_moving = True
    mock_idasen_desk.stop.side_effect = BleakError()

    await desk.move_to(50)
    mock_idasen_desk.move_to_target.assert_not_called()


async def test_move_fallback_without_monitoring(mock_idasen_desk: MagicMock):
    """Test that move_to falls back to move_to_target when monitor_height=False."""
    desk = Desk(None, False)
    await desk.connect(FAKE_BLE_DEVICE)

    await desk.move_to(50)
    mock_idasen_desk.wakeup.assert_awaited()
    mock_idasen_desk.move_to_target.assert_called_with(height_percent_to_meters(50))
    mock_idasen_desk._client.write_gatt_char.assert_not_called()


async def test_move_fallback_failure(mock_idasen_desk: MagicMock):
    """Test that move_to swallows BleakError from move_to_target."""
    desk = Desk(None, False)
    await desk.connect(FAKE_BLE_DEVICE)

    mock_idasen_desk.move_to_target.side_effect = BleakError()

    await desk.move_to(50)  # should not raise


async def test_move_skipped_when_at_target(mock_idasen_desk: MagicMock):
    """Test that move_to returns early without moving if already at target."""
    desk = await connect_at(mock_idasen_desk, 50)
    mock_idasen_desk.reset_mock()

    await desk.move_to(50)
    mock_idasen_desk.wakeup.assert_not_awaited()
    mock_idasen_desk.stop.assert_not_called()
    mock_idasen_desk._client.write_gatt_char.assert_not_called()


async def test_move_skipped_when_height_unknown(mock_idasen_desk: MagicMock):
    """Test that move_to does nothing if current height is not yet known."""
    desk = await connect_at(mock_idasen_desk, None)
    assert desk.height is None
    mock_idasen_desk.reset_mock()  # clear calls from connect (pair/wakeup)

    await desk.move_to(50)
    mock_idasen_desk.wakeup.assert_not_awaited()
    mock_idasen_desk.stop.assert_not_called()
    mock_idasen_desk._client.write_gatt_char.assert_not_called()


async def test_move_prepare_failure(mock_idasen_desk: MagicMock):
    """Test that move_to aborts if waking the desk fails."""
    desk = await connect_at(mock_idasen_desk, 0)
    mock_idasen_desk.wakeup.side_effect = BleakError()

    await desk.move_to(50)
    mock_idasen_desk._client.write_gatt_char.assert_not_called()


@pytest.mark.parametrize(
    "error", [BleakError(), BleakCharacteristicNotFoundError(_UUID_REFERENCE_INPUT)]
)
async def test_move_write_failure(mock_idasen_desk: MagicMock, error: Exception):
    """Test that a failed position write aborts the move without raising."""
    desk = await connect_at(mock_idasen_desk, 0)
    mock_idasen_desk._client.write_gatt_char.side_effect = error

    await desk.move_to(50)
    mock_idasen_desk._client.write_gatt_char.assert_awaited_once()


async def test_move_times_out(
    mock_idasen_desk: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """Test that move_to gives up if the target is never reached."""
    monkeypatch.setattr("idasen_ha._MOVE_TIMEOUT_S", 0.05)
    desk = await connect_at(mock_idasen_desk, 0)

    await desk.move_to(50)
    assert mock_idasen_desk._client.write_gatt_char.await_count > 1


async def test_stop_cancels_move(mock_idasen_desk: MagicMock):
    """Test that stop() ends the move loop before stopping the desk."""
    desk = await connect_at(mock_idasen_desk, 0)
    client = mock_idasen_desk._client

    move = asyncio.create_task(desk.move_to(50))  # never reaches target
    while client.write_gatt_char.await_count < 2:
        await asyncio.sleep(0)
    mock_idasen_desk.stop.reset_mock()

    await desk.stop()
    mock_idasen_desk.stop.assert_awaited_once()
    writes_at_stop = client.write_gatt_char.await_count

    await asyncio.sleep(0.05)
    assert client.write_gatt_char.await_count == writes_at_stop
    assert move.done()
    await move  # interrupted, not raised


async def test_new_move_cancels_previous(mock_idasen_desk: MagicMock):
    """Test that a new move_to replaces an in-progress one."""
    desk = await connect_at(mock_idasen_desk, 0)
    client = mock_idasen_desk._client

    first = asyncio.create_task(desk.move_to(100))  # never reaches target
    while not client.write_gatt_char.await_count:
        await asyncio.sleep(0)

    client.write_gatt_char.side_effect = reach_target_on_write(desk, 50)
    await desk.move_to(50)
    assert first.done()
    await first  # interrupted, not raised

    client.write_gatt_char.assert_awaited_with(
        _UUID_REFERENCE_INPUT,
        _meters_to_bytes(height_percent_to_meters(50)),
        response=False,
    )


async def test_cancelling_move_to_propagates(mock_idasen_desk: MagicMock):
    """Test that cancelling the caller of move_to cancels the move and re-raises."""
    desk = await connect_at(mock_idasen_desk, 0)
    client = mock_idasen_desk._client

    move = asyncio.create_task(desk.move_to(50))  # never reaches target
    while not client.write_gatt_char.await_count:
        await asyncio.sleep(0)

    move.cancel()
    with pytest.raises(asyncio.CancelledError):
        await move
    assert desk._move_task is None


async def test_stop(mock_idasen_desk: MagicMock):
    """Test stop call."""
    desk = Desk(None, False)
    await desk.connect(FAKE_BLE_DEVICE)

    await desk.stop()
    mock_idasen_desk.stop.assert_called()


@pytest.mark.parametrize("connect_first", [True, False])
async def test_no_ops_if_not_connected(
    mock_idasen_desk: MagicMock, connect_first: bool
):
    """Test that disconnect is called if pair fails."""
    desk = Desk(Mock(), False)

    if connect_first:
        await desk.connect(FAKE_BLE_DEVICE)
        mock_idasen_desk.is_connected = False

    assert not desk.is_connected
    await desk.move_to(100)
    await desk.stop()

    mock_idasen_desk.move_to_target.assert_not_called()
    mock_idasen_desk.stop.assert_not_called()


async def test_monitoring_skipped_if_disconnected_during_get_height(
    mock_idasen_desk: MagicMock,
):
    """Test that monitoring is skipped if the desk disconnects during get_height."""

    async def get_height_and_disconnect():
        # Simulate a BLE disconnect occurring while get_height is in progress
        mock_idasen_desk.is_connected = False
        return height_percent_to_meters(50)

    mock_idasen_desk.get_height.side_effect = get_height_and_disconnect

    desk = Desk(Mock(), True)
    await desk.connect(FAKE_BLE_DEVICE)

    mock_idasen_desk.monitor.assert_not_called()
